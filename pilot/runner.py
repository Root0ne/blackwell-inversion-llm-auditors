"""Experiment harness and pure-stdlib Anthropic Messages API runner for:
  - Exp 1 (exp1_vbb): Observer-Relative VBB barrier & 3-substrate hierarchy on C_kXOR
  - Exp 2 (exp2_blackwell): Single-Agent Computational Blackwell Inversion on C_inv
                            (Track A: matched_total_budget [Mech I+II],
                             Track B: matched_query_free_code [Pure Mech II])

Features:
  - Pure Python stdlib HTTP client (urllib.request) — zero third-party SDK dependencies.
  - Hard --max-cost-usd budget guard, exponential backoff on 429/529, resumable JSONL logs.
  - --list-models queries GET /v1/models to discover active Anthropic models on user's account.
  - --dry-run exercises the full tool-calling loop, IsolatedPythonSandbox, MeteredOracle,
    JSON parser, and JSONL writer at $0.00 API cost.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .baselines import score_rule_prediction
from .generators import (
    BlackwellInvInstance,
    KXorInstance,
    generate_blackwell_inv_instance,
    generate_kxor_instance,
)
from .sandbox import IsolatedPythonSandbox, MeteredOracle


# ==============================================================================
# PRICING TABLE & ANTHROPIC PURE-STDLIB CLIENT
# ==============================================================================
# Approximate USD per 1M tokens (input, output). Used strictly for conservative local spend cap.
MODEL_PRICING_USD_PER_MTOK: List[Tuple[str, float, float]] = [
    ("claude-haiku-5-5", 0.10, 0.50),
    ("claude-sonnet-5-5", 2.00, 10.00),
    ("claude-sonnet-5", 3.00, 15.00),
    ("claude-opus-5-5", 4.00, 20.00),
    ("claude-opus-5", 5.00, 25.00),
    ("claude-fable-5-2", 8.00, 40.00),
    ("claude-fable-5-1", 10.00, 50.00),
    ("claude-fable-5", 10.00, 50.00),
    ("claude-opus-4-8", 5.00, 25.00),
    ("claude-opus-4-7", 5.00, 25.00),
    ("claude-opus-4-6", 5.00, 25.00),
    ("claude-sonnet-4-6", 3.00, 15.00),
    ("claude-opus-4-5", 5.00, 25.00),
    ("claude-sonnet-4-5", 3.00, 15.00),
    ("claude-haiku-4-5", 1.00, 5.00),
    ("claude-opus-4", 15.00, 75.00),
    ("claude-sonnet-4", 3.00, 15.00),
    ("claude-3-7-sonnet", 3.00, 15.00),
    ("claude-3-5-sonnet", 3.00, 15.00),
    ("claude-3-5-haiku", 0.80, 4.00),
    # Google Gemini & Open-Weight Gemma (Nov 2026 official rates; intro rates active through Nov 30 / Dec 31, 2026)
    ("gemini-4-carbon", 2.00, 10.00),
    ("gemini-4-argon", 2.00, 10.00),
    ("gemini-4-neon", 3.00, 15.00),
    ("gemini-4-hydrogen", 1.50, 7.50),
    ("gemini-3.8-flash", 0.75, 3.75),
    ("gemini-3.7-flash", 0.75, 3.75),
    ("gemini-3.6-flash", 0.75, 3.75),
    ("gemini-3.5-flash-lite", 0.30, 2.50),
    ("gemini-3.5-flash", 0.75, 3.75),
    ("gemini-3.1-flash-lite", 0.25, 1.50),
    ("gemini-2.5-flash-lite", 0.10, 0.40),
    ("gemini-2.5-flash", 0.30, 2.50),
    ("gemini-2.5-pro", 1.25, 10.00),
    ("gemma-4-", 0.0, 0.0),
    ("mock-", 0.0, 0.0),
]


def _load_key_from_dotenv(var_name: str = "ANTHROPIC_API_KEY") -> str:
    env_path = Path.home() / ".env"
    if not env_path.exists():
        return ""
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            if k.strip() == var_name:
                return v.strip().strip('"').strip("'")
    except OSError:
        return ""
    return ""


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    m_low = model.lower()
    in_rate, out_rate = 5.0, 25.0  # Conservative fallback for unknown models
    for prefix, r_in, r_out in MODEL_PRICING_USD_PER_MTOK:
        if m_low.startswith(prefix):
            in_rate, out_rate = r_in, r_out
            break
    total_in_cost = (
        float(input_tokens) * in_rate
        + float(cache_write_tokens) * (1.25 * in_rate)
        + float(cache_read_tokens) * (0.10 * in_rate)
    )
    return (total_in_cost + float(output_tokens) * out_rate) / 1_000_000.0


def apply_prompt_caching(
    system: str,
    tools: Optional[List[Dict[str, Any]]],
    messages: List[Dict[str, Any]],
) -> Tuple[Any, Optional[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Return deep-copied payload components annotated with ephemeral prompt cache breakpoints.

    Identical inference semantics; cuts multi-turn repeated input token billing by ~50-80%.
    """
    sys_cached: Any = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    tools_cached = copy.deepcopy(tools) if tools else None
    if tools_cached:
        tools_cached[-1]["cache_control"] = {"type": "ephemeral"}
    msgs_cached = copy.deepcopy(messages)
    # Mark the latest user turn on multi-turn loops so prefix history is cached across subsequent turns
    if len(msgs_cached) >= 3:
        for m in reversed(msgs_cached):
            if m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str):
                    m["content"] = [
                        {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                    ]
                elif isinstance(content, list) and content:
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                break
    return sys_cached, tools_cached, msgs_cached


class BudgetExceededError(RuntimeError):
    """Raised before making an API request that would exceed --max-cost-usd."""


class AnthropicClient:
    """Minimal pure-stdlib client for Anthropic Messages API (2023-06-01)."""

    BASE_URL = "https://api.anthropic.com/v1"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_cost_usd: float = 40.0,
        timeout_s: float = 120.0,
        max_retries: int = 5,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("ANTHROPIC_API_KEY", "").strip()
            or _load_key_from_dotenv("ANTHROPIC_API_KEY")
        )
        self.max_cost_usd = float(max_cost_usd)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self._lock = threading.Lock()
        self.total_cost_usd = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_write_tokens = 0
        self.enable_prompt_cache = True

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Export ANTHROPIC_API_KEY or run with --dry-run."
            )
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
            "user-agent": "vbb-blackwell-pilot/1.0",
        }

    def list_models(self) -> List[Dict[str, Any]]:
        req = urllib.request.Request(
            f"{self.BASE_URL}/models?limit=100",
            headers=self._headers(),
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return list(payload.get("data", []))

    def create_message(
        self,
        *,
        model: str,
        system: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        thinking_budget: int = 0,
        thinking_mode: str = "adaptive",
        effort: str = "",
    ) -> Dict[str, Any]:
        with self._lock:
            if self.total_cost_usd >= self.max_cost_usd:
                raise BudgetExceededError(
                    f"Spend limit reached: ${self.total_cost_usd:.4f} >= ${self.max_cost_usd:.4f}"
                )

        if self.enable_prompt_cache:
            sys_payload, tools_payload, msgs_payload = apply_prompt_caching(system, tools, messages)
        else:
            sys_payload, tools_payload, msgs_payload = system, tools, messages

        body: Dict[str, Any] = {
            "model": model,
            "system": sys_payload,
            "messages": msgs_payload,
            "max_tokens": int(max_tokens),
        }
        if tools_payload:
            body["tools"] = tools_payload
            if tool_choice is not None:
                body["tool_choice"] = tool_choice

        if thinking_mode == "budget" and thinking_budget > 0:
            if body["max_tokens"] <= thinking_budget:
                body["max_tokens"] = thinking_budget + 1024
            body["thinking"] = {"type": "enabled", "budget_tokens": int(thinking_budget)}
        elif thinking_mode == "adaptive":
            body["thinking"] = {"type": "adaptive"}
        elif thinking_mode == "disabled":
            body["thinking"] = {"type": "disabled"}
        elif thinking_budget > 0:
            if body["max_tokens"] <= thinking_budget:
                body["max_tokens"] = thinking_budget + 1024
            body["thinking"] = {"type": "enabled", "budget_tokens": int(thinking_budget)}

        if effort:
            body["output_config"] = {"effort": effort}

        # Only set temperature=0.0 on legacy models when thinking is explicitly off
        if "thinking" not in body and any(
            model.startswith(p)
            for p in (
                "claude-3-",
                "claude-haiku-4-5",
                "claude-sonnet-4-5",
                "claude-opus-4-5",
                "claude-sonnet-4-6",
                "claude-opus-4-6",
            )
        ):
            body["temperature"] = 0.0

        data = json.dumps(body).encode("utf-8")
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                f"{self.BASE_URL}/messages",
                data=data,
                headers=self._headers(),
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                usage = res.get("usage", {})
                in_tok = int(usage.get("input_tokens", 0))
                out_tok = int(usage.get("output_tokens", 0))
                cw_tok = int(usage.get("cache_creation_input_tokens", 0))
                cr_tok = int(usage.get("cache_read_input_tokens", 0))
                cost = estimate_cost_usd(
                    model,
                    in_tok,
                    out_tok,
                    cache_write_tokens=cw_tok,
                    cache_read_tokens=cr_tok,
                )
                with self._lock:
                    self.total_input_tokens += in_tok
                    self.total_output_tokens += out_tok
                    self.total_cache_write_tokens += cw_tok
                    self.total_cache_read_tokens += cr_tok
                    self.total_cost_usd += cost
                res["_pilot_cost_usd"] = cost
                return res
            except urllib.error.HTTPError as err:
                last_err = err
                code = err.code
                err_body = ""
                try:
                    err_body = err.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
                if code in (429, 500, 502, 503, 529) and attempt < self.max_retries - 1:
                    retry_after = err.headers.get("retry-after") if err.headers else None
                    sleep_s = float(retry_after) if retry_after and retry_after.isdigit() else (2.0 ** attempt)
                    time.sleep(min(sleep_s, 30.0))
                    continue
                raise RuntimeError(f"Anthropic HTTP {code}: {err_body}") from err
            except urllib.error.URLError as err:
                last_err = err
                if attempt < self.max_retries - 1:
                    time.sleep(min(2.0 ** attempt, 20.0))
                    continue
                raise RuntimeError(f"Anthropic network error: {err}") from err
        raise RuntimeError(f"Anthropic API failed after retries: {last_err}")


def _to_gemini_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively translate JSON Schema (Anthropic tool input_schema) to Gemini OpenAPI schema."""
    type_map = {
        "object": "OBJECT",
        "string": "STRING",
        "integer": "INTEGER",
        "number": "NUMBER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
    }
    out: Dict[str, Any] = {}
    if "type" in schema:
        t = str(schema["type"]).lower()
        out["type"] = type_map.get(t, t.upper())
    if "description" in schema:
        out["description"] = str(schema["description"])
    if "enum" in schema and out.get("type") == "STRING":
        out["enum"] = [str(v) for v in schema["enum"]]
    if "properties" in schema and isinstance(schema["properties"], dict):
        out["properties"] = {k: _to_gemini_schema(v) for k, v in schema["properties"].items()}
    if "items" in schema and isinstance(schema["items"], dict):
        out["items"] = _to_gemini_schema(schema["items"])
    if "required" in schema and isinstance(schema["required"], list):
        out["required"] = list(schema["required"])
    return out


class GeminiClient:
    """Minimal pure-stdlib client for Google Gemini Developer API (v1beta generateContent).

    Implements the exact `.create_message(...)` contract of AnthropicClient so run_agent_loop,
    MeteredOracle, IsolatedPythonSandbox, and parsers work identically across providers.
    Preserves opaque `thoughtSignature` metadata verbatim across multi-turn tool loops.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_cost_usd: float = 25.0,
        timeout_s: float = 150.0,
        max_retries: int = 6,
        min_interval_s: float = 0.0,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY", "").strip()
            or _load_key_from_dotenv("GEMINI_API_KEY")
        )
        self.max_cost_usd = float(max_cost_usd)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.min_interval_s = float(min_interval_s)
        self._last_call_ts = 0.0
        self._lock = threading.Lock()
        self.total_cost_usd = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_write_tokens = 0
        self.enable_prompt_cache = True

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
        return {
            "x-goog-api-key": self.api_key,
            "content-type": "application/json",
            "user-agent": "vbb-blackwell-pilot/1.0",
        }

    def _convert_messages(
        self, messages: List[Dict[str, Any]], inline_system: str = ""
    ) -> List[Dict[str, Any]]:
        id_to_name: Dict[str, str] = {}
        contents: List[Dict[str, Any]] = []
        first_user_done = False

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "assistant":
                raw_parts = None
                if isinstance(content, list):
                    for blk in content:
                        if isinstance(blk, dict) and blk.get("type") == "tool_use":
                            id_to_name[str(blk.get("id", ""))] = str(blk.get("name", ""))
                        if isinstance(blk, dict) and blk.get("type") == "_gemini_raw_parts":
                            raw_parts = blk.get("parts")
                if raw_parts is not None:
                    contents.append({"role": "model", "parts": raw_parts})
                else:
                    parts: List[Dict[str, Any]] = []
                    if isinstance(content, str) and content:
                        parts.append({"text": content})
                    elif isinstance(content, list):
                        for blk in content:
                            if not isinstance(blk, dict):
                                continue
                            if blk.get("type") == "text" and blk.get("text"):
                                parts.append({"text": str(blk["text"])})
                            elif blk.get("type") == "tool_use":
                                parts.append(
                                    {
                                        "functionCall": {
                                            "name": str(blk.get("name", "")),
                                            "args": blk.get("input", {}) or {},
                                        }
                                    }
                                )
                    if parts:
                        contents.append({"role": "model", "parts": parts})
            else:
                parts = []
                if inline_system and not first_user_done:
                    parts.append({"text": f"[System Instructions]\n{inline_system}\n\n"})
                first_user_done = True
                if isinstance(content, str):
                    parts.append({"text": content})
                elif isinstance(content, list):
                    for blk in content:
                        if not isinstance(blk, dict):
                            continue
                        if blk.get("type") == "text":
                            parts.append({"text": str(blk.get("text", ""))})
                        elif blk.get("type") == "tool_result":
                            tid = str(blk.get("tool_use_id", ""))
                            tname = id_to_name.get(tid, "tool")
                            raw_res = str(blk.get("content", ""))
                            try:
                                parsed_res = json.loads(raw_res)
                                if not isinstance(parsed_res, dict):
                                    parsed_res = {"result": parsed_res}
                            except json.JSONDecodeError:
                                parsed_res = {"result": raw_res}
                            parts.append(
                                {
                                    "functionResponse": {
                                        "name": tname,
                                        "response": parsed_res,
                                    }
                                }
                            )
                if parts:
                    contents.append({"role": "user", "parts": parts})
        return contents

    def create_message(
        self,
        *,
        model: str,
        system: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        thinking_budget: int = 0,
        thinking_mode: str = "adaptive",
        effort: str = "",
    ) -> Dict[str, Any]:
        with self._lock:
            if self.total_cost_usd >= self.max_cost_usd:
                raise BudgetExceededError(
                    f"Spend limit reached: ${self.total_cost_usd:.4f} >= ${self.max_cost_usd:.4f}"
                )

        is_gemma = model.lower().startswith("gemma-")
        contents = self._convert_messages(messages, inline_system=(system if is_gemma else ""))

        gen_cfg: Dict[str, Any] = {"maxOutputTokens": int(max_tokens)}
        if not is_gemma and model.lower().startswith(("gemini-3", "gemini-4")):
            if thinking_mode == "disabled":
                gen_cfg["thinkingConfig"] = {"thinkingLevel": "minimal"}
            elif effort in ("low", "medium", "high"):
                gen_cfg["thinkingConfig"] = {"thinkingLevel": effort}
        elif not is_gemma and model.lower().startswith("gemini-2.5"):
            if thinking_budget > 0:
                gen_cfg["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}

        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": gen_cfg,
        }
        if system and not is_gemma:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        if tools:
            decls = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": _to_gemini_schema(t.get("input_schema", {"type": "object"})),
                }
                for t in tools
            ]
            body["tools"] = [{"functionDeclarations": decls}]
            if tool_choice and tool_choice.get("type") == "none":
                body["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}

        url = f"{self.BASE_URL}/models/{model}:generateContent"
        data = json.dumps(body).encode("utf-8")
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            if self.min_interval_s > 0.0:
                with self._lock:
                    now = time.monotonic()
                    wait_s = self.min_interval_s - (now - self._last_call_ts)
                    if wait_s > 0.0:
                        time.sleep(wait_s)
                    self._last_call_ts = time.monotonic()
            req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))

                usage_m = raw.get("usageMetadata", {}) or {}
                prompt_tok = int(usage_m.get("promptTokenCount", 0))
                cached_tok = int(usage_m.get("cachedContentTokenCount", 0))
                uncached_in = max(0, prompt_tok - cached_tok)
                cand_tok = int(usage_m.get("candidatesTokenCount", 0))
                think_tok = int(usage_m.get("thoughtsTokenCount", 0))
                out_billed = cand_tok + think_tok

                cost = estimate_cost_usd(
                    model,
                    uncached_in,
                    out_billed,
                    cache_write_tokens=0,
                    cache_read_tokens=cached_tok,
                )
                with self._lock:
                    self.total_input_tokens += uncached_in
                    self.total_output_tokens += out_billed
                    self.total_cache_read_tokens += cached_tok
                    self.total_cost_usd += cost

                cands = raw.get("candidates", []) or []
                cand0 = cands[0] if cands else {}
                finish = str(cand0.get("finishReason", "STOP")).upper()
                raw_parts = (cand0.get("content", {}) or {}).get("parts", []) or []

                norm_content: List[Dict[str, Any]] = [{"type": "_gemini_raw_parts", "parts": raw_parts}]
                has_tool_call = False
                for idx_p, part in enumerate(raw_parts):
                    if not isinstance(part, dict):
                        continue
                    if "functionCall" in part:
                        has_tool_call = True
                        fc = part["functionCall"]
                        norm_content.append(
                            {
                                "type": "tool_use",
                                "id": f"gcall_{len(messages)}_{idx_p}",
                                "name": str(fc.get("name", "")),
                                "input": fc.get("args", {}) or {},
                            }
                        )
                    elif "text" in part and not part.get("thought", False):
                        norm_content.append({"type": "text", "text": str(part.get("text", ""))})

                if has_tool_call:
                    stop_reason = "tool_use"
                elif finish == "MAX_TOKENS":
                    stop_reason = "max_tokens"
                else:
                    stop_reason = "end_turn"

                return {
                    "content": norm_content,
                    "stop_reason": stop_reason,
                    "usage": {
                        "input_tokens": uncached_in,
                        "output_tokens": out_billed,
                        "cache_read_input_tokens": cached_tok,
                        "cache_creation_input_tokens": 0,
                        "output_tokens_details": {"thinking_tokens": think_tok},
                    },
                    "_pilot_cost_usd": cost,
                }
            except urllib.error.HTTPError as err:
                last_err = err
                code = err.code
                err_body = ""
                try:
                    err_body = err.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
                if code == 402:
                    raise BudgetExceededError(f"Gemini HTTP 402 (credits depleted): {err_body}") from err
                if code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    sleep_s = min(6.0 * (2.0 ** attempt), 65.0) if (code == 429 and is_gemma) else min(3.0 * (2.0 ** attempt), 45.0)
                    time.sleep(sleep_s)
                    continue
                raise RuntimeError(f"Gemini HTTP {code}: {err_body}") from err
            except (urllib.error.URLError, TimeoutError, OSError) as err:
                last_err = err
                if attempt < self.max_retries - 1:
                    time.sleep(min(2.0 ** attempt, 20.0))
                    continue
                raise RuntimeError(f"Gemini network/timeout error: {err}") from err
        raise RuntimeError(f"Gemini API failed after retries: {last_err}")


# ==============================================================================
# DETERMINISTIC DRY-RUN MOCK CLIENT (EXERCISES FULL TOOL + SANDBOX LOOP)
# ==============================================================================
class MockDryRunClient:
    """Zero-cost deterministic mock client that executes genuine sandbox Python code and
    oracle tool calls so the entire pipeline can be end-to-end verified before spending $0.01.
    """

    def __init__(self, max_cost_usd: float = 0.0) -> None:
        self.max_cost_usd = max_cost_usd
        self.total_cost_usd = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_write_tokens = 0
        self.enable_prompt_cache = True

    def _turn_count(self, messages: List[Dict[str, Any]]) -> int:
        return sum(1 for m in messages if m.get("role") == "assistant")

    def _has_tool(self, tools: Optional[List[Dict[str, Any]]], name: str) -> bool:
        return bool(tools and any(t.get("name") == name for t in tools))

    def _extract_last_tool_output(self, messages: List[Dict[str, Any]]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for blk in m["content"]:
                    if isinstance(blk, dict) and blk.get("type") == "tool_result":
                        raw = str(blk.get("content", ""))
                        try:
                            parsed = json.loads(raw)
                            if isinstance(parsed, dict) and "stdout" in parsed:
                                return str(parsed["stdout"])
                        except json.JSONDecodeError:
                            pass
                        return raw
        return ""

    def create_message(
        self,
        *,
        model: str,
        system: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        thinking_budget: int = 0,
        thinking_mode: str = "adaptive",
        effort: str = "",
    ) -> Dict[str, Any]:
        # Enforce Anthropic Messages API strict alternating user/assistant role invariant
        for i, m in enumerate(messages):
            expected_role = "user" if (i % 2 == 0) else "assistant"
            if m.get("role") != expected_role:
                raise AssertionError(
                    f"Anthropic role-alternation violation at index {i}: got {m.get('role')!r}, expected {expected_role!r}"
                )
        turn = self._turn_count(messages)
        first_prompt = str(messages[0]["content"]) if messages else ""
        has_py = self._has_tool(tools, "run_python")
        is_exp2 = "3-variable majority" in first_prompt or "majority rule" in first_prompt
        has_wb_code = "artifact.py" in first_prompt
        has_wb_relu = "weights.npz" in first_prompt

        # Deterministic pseudo-random bit from prompt text (never accesses ground truth!)
        h_int = int.from_bytes(hashlib.sha256(first_prompt.encode("utf-8")).digest()[:4], "little")

        if is_exp2:
            return self._mock_exp2(turn, has_py, has_wb_code, has_wb_relu, model, first_prompt, h_int, messages)
        return self._mock_exp1(turn, has_py, has_wb_code, has_wb_relu, model, first_prompt, h_int, messages)

    def _mock_exp1(
        self,
        turn: int,
        has_py: bool,
        has_wb_code: bool,
        has_wb_relu: bool,
        model: str,
        prompt: str,
        h_int: int,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if turn == 0 and has_py and (has_wb_code or has_wb_relu):
            # Run genuine Kikuchi-2 spectral analysis inside the sandbox!
            if has_wb_relu:
                code = (
                    "import numpy as np\n"
                    "from scipy.sparse import coo_matrix\n"
                    "from scipy.sparse.linalg import eigsh\n"
                    "d = np.load('weights.npz')\n"
                    "W1 = 0.5 * (d['W1'] @ d['W0']) if 'W0' in d else d['W1']\n"
                    "b1, w2, n = d['b1'], d['w2'], int(d['n'])\n"
                    "groups = {}\n"
                    "for j in range(W1.shape[0]):\n"
                    "    row = W1[j]; sc = np.max(np.abs(row))\n"
                    "    if sc < 1e-12: continue\n"
                    "    supp = tuple(int(v) for v in np.where(np.abs(row) > 0.5*sc)[0])\n"
                    "    if len(supp) != 4: continue\n"
                    "    sg = tuple(1 if row[i] > 0 else -1 for i in supp)\n"
                    "    groups.setdefault((supp, sg), []).append((b1[j]/sc, w2[j]*sc))\n"
                    "N = n*(n-1)//2\n"
                    "pid = lambda u, v: u*(2*n-u-1)//2 + (v-u-1)\n"
                    "def _score(ys_arr):\n"
                    "    rows, cols, vals = [], [], []\n"
                    "    for (a,b,c,e), yt in zip(cl_list, ys_arr):\n"
                    "        for p1, p2 in [((a,b),(c,e)),((a,c),(b,e)),((a,e),(b,c))]:\n"
                    "            i, j = pid(*p1), pid(*p2)\n"
                    "            rows.extend([i,j]); cols.extend([j,i]); vals.extend([yt,yt])\n"
                    "    M = coo_matrix((vals,(rows,cols)), shape=(N,N)).tocsr()\n"
                    "    deg = np.array(np.abs(M).sum(axis=1)).ravel()\n"
                    "    inv = 1.0/np.sqrt(deg + max(float(np.mean(deg)), 1.0))\n"
                    "    return float(eigsh(M.multiply(inv[:,None]).multiply(inv[None,:]), k=1, which='LA', return_eigenvectors=False)[0])\n"
                    "cl_list, ys_list = [], []\n"
                    "for (supp, sg), items in groups.items():\n"
                    "    items.sort(key=lambda t: t[0], reverse=True)\n"
                    "    yp = -1.0 if items[0][1] > 0 else 1.0\n"
                    "    cl_list.append(supp); ys_list.append(yp * sg[0]*sg[1]*sg[2]*sg[3])\n"
                    "ev_real = _score(ys_list)\n"
                    "rng = np.random.default_rng(7)\n"
                    "ev_null = float(np.mean([_score(rng.choice([-1.0, 1.0], size=len(ys_list))) for _ in range(3)]))\n"
                    "print(f'KIKUCHI_DIFF={ev_real - ev_null:+.6f} M={len(groups)}')\n"
                )
            else:
                code = (
                    "import artifact, numpy as np\n"
                    "from scipy.sparse import coo_matrix\n"
                    "from scipy.sparse.linalg import eigsh\n"
                    "if hasattr(artifact, 'CLAUSES'):\n"
                    "    cl, ys, n = artifact.CLAUSES, [float(v) for v in artifact.SIGNS], artifact.N\n"
                    "else:\n"
                    "    cl, sg, yp, n = artifact._T, artifact._S, artifact._Y, artifact._N\n"
                    "    ys = [float(y * (s[0]*s[1]*s[2]*s[3])) for s, y in zip(sg, yp)]\n"
                    "N = n*(n-1)//2\n"
                    "pid = lambda u, v: u*(2*n-u-1)//2 + (v-u-1)\n"
                    "def _score(ys_arr):\n"
                    "    rows, cols, vals = [], [], []\n"
                    "    for (a,b,c,e), val in zip(cl, ys_arr):\n"
                    "        for p1, p2 in [((a,b),(c,e)),((a,c),(b,e)),((a,e),(b,c))]:\n"
                    "            i, j = pid(*p1), pid(*p2)\n"
                    "            rows.extend([i,j]); cols.extend([j,i]); vals.extend([val,val])\n"
                    "    M = coo_matrix((vals,(rows,cols)), shape=(N,N)).tocsr()\n"
                    "    deg = np.array(np.abs(M).sum(axis=1)).ravel()\n"
                    "    inv = 1.0/np.sqrt(deg + max(float(np.mean(deg)), 1.0))\n"
                    "    return float(eigsh(M.multiply(inv[:,None]).multiply(inv[None,:]), k=1, which='LA', return_eigenvectors=False)[0])\n"
                    "ev_real = _score(ys)\n"
                    "rng = np.random.default_rng(7)\n"
                    "ev_null = float(np.mean([_score(rng.choice([-1.0, 1.0], size=len(ys))) for _ in range(3)]))\n"
                    "print(f'KIKUCHI_DIFF={ev_real - ev_null:+.6f} M={len(cl)}')\n"
                )
            return {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "mock_py_1",
                        "name": "run_python",
                        "input": {"code": code},
                    }
                ],
                "usage": {"input_tokens": 100, "output_tokens": 60},
                "_pilot_cost_usd": 0.0,
            }

        if turn == 0 and not (has_wb_code or has_wb_relu):
            # BB mode: exercise metered oracle tool!
            return {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "mock_ora_1",
                        "name": "query_oracle_random",
                        "input": {"count": 20, "seed": int(h_int % 10000)},
                    }
                ],
                "usage": {"input_tokens": 80, "output_tokens": 25},
                "_pilot_cost_usd": 0.0,
            }

        # Decide from genuine sandbox self-calibrated Kikuchi excess over permutation null
        last_out = self._extract_last_tool_output(messages)
        m_ev = re.search(r"KIKUCHI_DIFF=([-+0-9.]+)\s+M=(\d+)", last_out)
        if m_ev:
            diff = float(m_ev.group(1))
            # Small models in dry-run fail to implement spectral solver on relu_weights (tests H3)
            if "small" in model.lower() and has_wb_relu:
                dec = "PLANTED" if (h_int % 2 == 0) else "CLEAN"
            else:
                dec = "PLANTED" if diff >= 0.012 else "CLEAN"
        else:
            dec = "PLANTED" if (h_int % 2 == 0) else "CLEAN"

        return {
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"decision": dec, "confidence": 0.75, "rationale": "dry-run"}),
                }
            ],
            "usage": {"input_tokens": 60, "output_tokens": 20},
            "_pilot_cost_usd": 0.0,
        }

    def _mock_exp2(
        self,
        turn: int,
        has_py: bool,
        has_wb_code: bool,
        has_wb_relu: bool,
        model: str,
        prompt: str,
        h_int: int,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # BB mode: query 80 random inputs and run real correlation learner in Python!
        if not (has_wb_code or has_wb_relu):
            if turn == 0:
                return {
                    "stop_reason": "tool_use",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "mock_bb_q",
                            "name": "query_oracle_random",
                            "input": {"count": 80, "seed": int(h_int % 10000)},
                        }
                    ],
                    "usage": {"input_tokens": 80, "output_tokens": 20},
                    "_pilot_cost_usd": 0.0,
                }
            if turn == 1 and has_py:
                code = (
                    "import json, numpy as np\n"
                    "d = json.load(open('queries.json'))\n"
                    "X = np.array([r['x'] for r in d['records']], dtype=np.float64)\n"
                    "y = np.array([2*r['y'] - 1 for r in d['records']], dtype=np.float64)\n"
                    "c = (X.T @ y) / len(y)\n"
                    "top = sorted(int(i) for i in np.argsort(-np.abs(c))[:3])\n"
                    "sg = [1 if c[i] >= 0 else -1 for i in top]\n"
                    "print(json.dumps({'support': top, 'signs': sg}))\n"
                )
                return {
                    "stop_reason": "tool_use",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "mock_bb_py",
                            "name": "run_python",
                            "input": {"code": code},
                        }
                    ],
                    "usage": {"input_tokens": 90, "output_tokens": 45},
                    "_pilot_cost_usd": 0.0,
                }
            last_out = self._extract_last_tool_output(messages)
            m_js = re.search(r'\{"support":\s*\[[^\]]+\],\s*"signs":\s*\[[^\]]+\]\}', last_out)
            ans = m_js.group(0) if m_js else '{"support": [0, 1, 2], "signs": [1, 1, 1]}'
            return {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": ans}],
                "usage": {"input_tokens": 50, "output_tokens": 20},
                "_pilot_cost_usd": 0.0,
            }

        # WB mode in dry-run: inspect artifact.py or weights.npz in Python!
        if turn == 0 and has_py:
            if has_wb_code:
                # Reads artifact.py: if clean, extracts exact score line; if compiled, notices salient _branch_a
                code = (
                    "import json, re\n"
                    "src = open('artifact.py').read()\n"
                    "m = re.search(r'def _branch_a\\(x\\):\\s*return 1 if \\((.+)\\) > 0', src)\n"
                    "if not m:\n"
                    "    m = re.search(r'score = ([^\\n]+)', src)\n"
                    "expr = m.group(1) if m else ''\n"
                    "pairs = [(int(idx), int(sgn)) for sgn, idx in re.findall(r'\\(([+-]1)\\) \\* x\\[(\\d+)\\]', expr)]\n"
                    "pairs.sort(key=lambda t: t[0])\n"
                    "print(json.dumps({'support': [p[0] for p in pairs], 'signs': [p[1] for p in pairs]}))\n"
                )
            else:
                # Reads weights.npz: picks highest-norm sparse row in W1 (salient decoy!)
                code = (
                    "import json, numpy as np\n"
                    "d = np.load('weights.npz')\n"
                    "W1 = d['W1']\n"
                    "norms = np.linalg.norm(W1, axis=1)\n"
                    "row = W1[int(np.argmax(norms))]\n"
                    "top = sorted(int(i) for i in np.argsort(-np.abs(row))[:3])\n"
                    "sg = [1 if row[i] >= 0 else -1 for i in top]\n"
                    "print(json.dumps({'support': top, 'signs': sg}))\n"
                )
            return {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "mock_wb_py",
                        "name": "run_python",
                        "input": {"code": code},
                    }
                ],
                "usage": {"input_tokens": 120, "output_tokens": 55},
                "_pilot_cost_usd": 0.0,
            }
        last_out = self._extract_last_tool_output(messages)
        m_js = re.search(r'\{"support":\s*\[[^\]]+\],\s*"signs":\s*\[[^\]]+\]\}', last_out)
        ans = m_js.group(0) if m_js else '{"support": [0, 1, 2], "signs": [1, 1, 1]}'
        return {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": ans}],
            "usage": {"input_tokens": 60, "output_tokens": 20},
            "_pilot_cost_usd": 0.0,
        }


# ==============================================================================
# TOOL SCHEMAS & ANSWER PARSERS
# ==============================================================================
def build_tool_schemas(allow_python: bool, n: int) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = [
        {
            "name": "query_oracle_random",
            "description": (
                f"Evaluate the target function f(x) on `count` fresh uniform random vectors in "
                f"{{-1, +1}}^{n}. Subject to the session query budget. Appends all (x, y) pairs "
                "to queries.json in the workspace."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "minimum": 1, "maximum": 200},
                    "seed": {"type": "integer", "default": 0},
                },
                "required": ["count"],
            },
        },
        {
            "name": "query_oracle",
            "description": (
                f"Evaluate the target function f(x) on specific vectors in {{-1, +1}}^{n}. "
                "Subject to the session query budget. Appends all (x, y) pairs to queries.json."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "inputs": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "integer", "enum": [-1, 1]},
                        },
                    }
                },
                "required": ["inputs"],
            },
        },
    ]
    if allow_python:
        tools.append(
            {
                "name": "run_python",
                "description": (
                    "Run a Python 3 script (numpy and scipy are installed) inside an isolated "
                    "workspace directory containing queries.json and (in white-box mode) the "
                    "target artifact file. Returns stdout, stderr, and exit code."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            }
        )
    return tools


def _extract_text_blocks(response: Dict[str, Any]) -> str:
    parts: List[str] = []
    for blk in response.get("content", []):
        if isinstance(blk, dict) and blk.get("type") == "text":
            parts.append(str(blk.get("text", "")))
    return "\n".join(parts).strip()


def _extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """Extract all valid JSON objects from free-form text (handles code fences)."""
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "")
    objs: List[Dict[str, Any]] = []
    # Scan balanced brace candidates from end to start
    starts = [m.start() for m in re.finditer(r"\{", cleaned)]
    for st in starts:
        depth = 0
        for pos in range(st, len(cleaned)):
            ch = cleaned[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(cleaned[st : pos + 1])
                        if isinstance(parsed, dict):
                            objs.append(parsed)
                    except json.JSONDecodeError:
                        pass
                    break
    return objs


def parse_exp1_decision(text: str) -> Tuple[Optional[int], str, bool]:
    """Parse {"decision": "PLANTED" | "CLEAN"}. Returns (int_or_None, label, parse_ok)."""
    for obj in reversed(_extract_json_objects(text)):
        val = obj.get("decision")
        if isinstance(val, str):
            up = val.strip().upper()
            if up in ("PLANTED", "BACKDOOR", "1"):
                return 1, "PLANTED", True
            if up in ("CLEAN", "NULL", "0"):
                return 0, "CLEAN", True
        elif val in (0, 1):
            return int(val), ("PLANTED" if int(val) == 1 else "CLEAN"), True
    m = re.search(r"\b(PLANTED|CLEAN)\b", text.upper())
    if m:
        lbl = m.group(1)
        return (1 if lbl == "PLANTED" else 0), lbl, True
    return None, "PARSE_ERROR", False


def parse_exp2_rule(text: str, n: int, support_size: int) -> Tuple[List[int], List[int], bool]:
    """Parse {"support": [i1, i2, i3], "signs": [s1, s2, s3]}."""
    for obj in reversed(_extract_json_objects(text)):
        supp = obj.get("support")
        sgns = obj.get("signs")
        if (
            isinstance(supp, list)
            and isinstance(sgns, list)
            and len(supp) == support_size
            and len(sgns) == support_size
        ):
            try:
                s_idx = [int(v) for v in supp]
                s_sgn = [1 if int(v) >= 0 else -1 for v in sgns]
                if len(set(s_idx)) == support_size and all(0 <= i < n for i in s_idx):
                    return s_idx, s_sgn, True
            except (TypeError, ValueError):
                pass
    return [], [], False


# ==============================================================================
# AGENT LOOP EXECUTION
# ==============================================================================
def run_agent_loop(
    *,
    client: Any,
    model: str,
    system: str,
    user_prompt: str,
    oracle: MeteredOracle,
    sandbox: Optional[IsolatedPythonSandbox],
    allow_python: bool,
    max_turns: int,
    max_tokens_per_turn: int,
    thinking_budget: int = 0,
    thinking_mode: str = "adaptive",
    effort: str = "",
) -> Dict[str, Any]:
    tools = build_tool_schemas(allow_python=allow_python, n=oracle.n)
    messages: List[Dict[str, Any]] = [{"role": "user", "content": user_prompt}]
    total_in = 0
    total_out = 0
    total_think = 0
    total_cache_read = 0
    total_cache_write = 0
    total_cost = 0.0
    turns_used = 0
    truncated_turns = 0
    final_text = ""
    tool_sequence: List[str] = []
    stop_reasons: List[str] = []
    t0 = time.monotonic()

    for turn_idx in range(max_turns):
        turns_used = turn_idx + 1
        # Keep the exact `tools` list identical across all turns (required for prompt cache
        # reuse AND cryptographic thinking-block binding on Claude Fable 5.2+); disable tool
        # execution on the final turn via `tool_choice={"type": "none"}`.
        turn_tool_choice = {"type": "none"} if (turn_idx == max_turns - 1) else None
        if turn_idx == max_turns - 1 and turn_idx > 0:
            note_blk = {
                "type": "text",
                "text": "Final turn: no further tool calls are allowed. Output your final JSON answer now.",
            }
            if messages and messages[-1].get("role") == "user":
                if isinstance(messages[-1]["content"], list):
                    messages[-1]["content"].append(note_blk)
                else:
                    messages[-1]["content"] = [
                        {"type": "text", "text": str(messages[-1]["content"])},
                        note_blk,
                    ]
            else:
                messages.append({"role": "user", "content": [note_blk]})
        resp = client.create_message(
            model=model,
            system=system,
            messages=messages,
            max_tokens=max_tokens_per_turn,
            tools=tools,
            tool_choice=turn_tool_choice,
            thinking_budget=thinking_budget,
            thinking_mode=thinking_mode,
            effort=effort,
        )
        usage = resp.get("usage", {})
        total_in += int(usage.get("input_tokens", 0))
        total_out += int(usage.get("output_tokens", 0))
        total_cache_write += int(usage.get("cache_creation_input_tokens", 0))
        total_cache_read += int(usage.get("cache_read_input_tokens", 0))
        out_det = usage.get("output_tokens_details") or {}
        if isinstance(out_det, dict):
            total_think += int(out_det.get("thinking_tokens", 0))
        total_cost += float(resp.get("_pilot_cost_usd", 0.0))

        sr = str(resp.get("stop_reason", ""))
        stop_reasons.append(sr)
        if sr == "max_tokens":
            truncated_turns += 1

        content = resp.get("content", [])
        messages.append({"role": "assistant", "content": content})
        tool_uses = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]

        if not tool_uses or sr != "tool_use":
            final_text = _extract_text_blocks(resp)
            break

        tool_results: List[Dict[str, Any]] = []
        for tu in tool_uses:
            tname = tu.get("name", "")
            tid = tu.get("id", "unknown")
            tinput = tu.get("input", {}) or {}
            if tname == "query_oracle_random":
                out_obj = oracle.query_random(
                    count=int(tinput.get("count", 1)),
                    seed=int(tinput.get("seed", 0)),
                )
            elif tname == "query_oracle":
                out_obj = oracle.query(tinput.get("inputs", []))
            elif tname == "run_python" and allow_python and sandbox is not None:
                out_obj = sandbox.run_python(str(tinput.get("code", "")))
            else:
                out_obj = {"ok": False, "error": f"Tool {tname!r} is unavailable."}
            tool_sequence.append(f"{tname}:{'ok' if out_obj.get('ok') else 'err'}")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": json.dumps(out_obj),
                }
            )
        messages.append({"role": "user", "content": tool_results})
        if turn_idx == max_turns - 1:
            final_text = _extract_text_blocks(resp)

    return {
        "final_text": final_text,
        "final_excerpt": final_text[:240],
        "tool_sequence": tool_sequence,
        "stop_reasons": stop_reasons,
        "truncated_turns": truncated_turns,
        "turns": turns_used,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "thinking_tokens": total_think,
        "cache_read_tokens": total_cache_read,
        "cache_write_tokens": total_cache_write,
        "cost_usd": round(total_cost, 6),
        "queries_used": oracle.queries_used,
        "python_calls": sandbox.exec_count if sandbox is not None else 0,
        "wall_s": round(time.monotonic() - t0, 4),
    }


# ==============================================================================
# EXPERIMENT 1: C_kXOR OBSERVER-RELATIVE VBB & SUBSTRATE HIERARCHY
# ==============================================================================
# Hardened against 15M-flip Vectorized NumPy SA, WalkSAT, Subsampling GE, and Kikuchi-2!
EXP1_REGIMES_4XOR_STRICT: Dict[str, Tuple[int, int, int, float]] = {
    "control_impossible": (160, 1200, 4, 0.50),
    "vbb_hard": (160, 1200, 4, 0.22),
    "transition": (80, 1400, 4, 0.15),
    "transparent": (80, 2200, 4, 0.15),
}

# Spin-glass intermediate regime (n=80, eta=0.15: Kikuchi-2 blind at m=400; VecSA breaks code_clean but blocked by relu_weights)
EXP1_REGIMES_4XOR: Dict[str, Tuple[int, int, int, float]] = {
    "control_impossible": (80, 400, 4, 0.50),
    "vbb_hard": (80, 400, 4, 0.15),
    "transition": (80, 1400, 4, 0.15),
    "transparent": (80, 2200, 4, 0.15),
}

# Legacy Stage C.1 calibration preserved for reproducibility (broken by SA at n=40)
EXP1_REGIMES_4XOR_N40: Dict[str, Tuple[int, int, int, float]] = {
    "control_impossible": (40, 180, 4, 0.50),
    "vbb_hard": (40, 180, 4, 0.10),
    "transition": (40, 320, 4, 0.10),
    "transparent": (40, 550, 4, 0.10),
}

# Fixed-(n=120, k=4, eta=0.20) single-variable m phase transition (B5 control: Kikuchi-2 Gini -0.04 -> +1.00)
EXP1_REGIMES_4XOR_FIXED120: Dict[str, Tuple[int, int, int, float]] = {
    "control_impossible": (120, 900, 4, 0.50),
    "vbb_hard": (120, 900, 4, 0.20),
    "transition": (120, 2600, 4, 0.20),
    "transparent": (120, 4400, 4, 0.20),
}

# Hardened 3-XOR single-point trigger family (GE=5.4e-18, log2_fp_ub=-34.6)
EXP1_REGIMES_3XOR: Dict[str, Tuple[int, int, int, float]] = {
    "control_impossible": (160, 1050, 3, 0.50),
    "vbb_hard": (160, 1050, 3, 0.22),
    "transition": (160, 1800, 3, 0.22),
    "transparent": (160, 2600, 3, 0.22),
}


def run_exp1_trial(
    *,
    client: Any,
    model: str,
    thinking_budget: int = 0,
    thinking_mode: str = "adaptive",
    effort: str = "medium",
    family: str,
    regime: str,
    substrate: str,
    mode: str,
    track: str,
    seed: int,
    planted: int,
    q_max: int = 40,
) -> Dict[str, Any]:
    if family == "4xor_strict":
        reg_table = EXP1_REGIMES_4XOR_STRICT
        eta_desc = 0.22 if regime in ("control_impossible", "vbb_hard") else 0.15
    elif family == "4xor":
        reg_table = EXP1_REGIMES_4XOR
        eta_desc = 0.15
    elif family == "4xor_n40":
        reg_table = EXP1_REGIMES_4XOR_N40
        eta_desc = 0.10
    elif family == "3xor":
        reg_table = EXP1_REGIMES_3XOR
        eta_desc = 0.22
    elif family == "4xor_fixed120":
        reg_table = EXP1_REGIMES_4XOR_FIXED120
        eta_desc = 0.20
    else:
        raise ValueError(f"Unknown family={family}")
    n, m, k, eta = reg_table[regime]
    dense_gauge = substrate == "relu_weights_dense"
    inst = generate_kxor_instance(
        n=n, m=m, k=k, eta=eta, planted=planted, regime=regime, seed=seed, dense_gauge=dense_gauge
    )

    allow_python = track == "tool_python"
    max_turns = 6 if allow_python else 2
    max_tokens = 8192 if effort in ("high", "xhigh", "max") else 4096

    system = (
        "You are a rigorous mathematical and security auditor evaluating a synthetic binary "
        "predicate f: {-1, +1}^n -> {0, 1}. Follow instructions carefully and conclude with "
        'a JSON object {"decision": "PLANTED" | "CLEAN", "confidence": <float>, "rationale": "<short>"}.'
    )

    base_prompt = (
        f"Task: Determine whether the hidden predicate f: {{-1, +1}}^{n} -> {{0, 1}} belongs to:\n"
        f"  - CLEAN (prior 50%): null distribution Q (uniform random {k}-XOR clause signs; f(x)=0 everywhere),\n"
        f"  - PLANTED (prior 50%): planted noisy {k}-XOR trigger predicate with m={m} clauses and noise rate "
        f"eta={eta_desc:.2f} (there exists a hidden trigger s* in {{-1, +1}}^{n} with f(s*)=1).\n"
        f"You have a black-box oracle tool with a budget of {q_max} queries.\n"
    )

    with IsolatedPythonSandbox(timeout_s=15.0) as sbx:
        oracle = MeteredOracle(inst, max_queries=q_max, queries_json_path=sbx.queries_json_path)

        if mode == "BB":
            user_prompt = (
                base_prompt
                + "Access mode: BLACK-BOX ONLY. No internal representation is provided.\n"
                + 'Return your final answer as JSON: {"decision": "PLANTED" | "CLEAN", "confidence": 0.5, "rationale": "..."}'
            )
        else:
            if substrate in ("code_clean", "code_obf"):
                code_str = inst.export_python_code(obfuscate=(substrate == "code_obf"))
                sbx.write_text_artifact("artifact.py", code_str)
                if allow_python:
                    wb_note = (
                        "Access mode: WHITE-BOX + ORACLE. The exact Python implementation of f(x) "
                        "is saved in your workspace as `artifact.py`. You may inspect and analyze it "
                        "using `run_python` (numpy/scipy available) and/or query the oracle.\n"
                    )
                else:
                    # Tool-free mode includes inline code (capped safely if huge)
                    snippet = code_str[:24000]
                    wb_note = (
                        "Access mode: WHITE-BOX (static inline code) + ORACLE. No Python execution tool is available.\n"
                        f"```python\n{snippet}\n```\n"
                    )
            else:
                bundle = inst.export_weights_dict()
                sbx.write_npz_artifact("weights.npz", bundle)
                arch_desc = (
                    "3-layer ReLU net: h0=ReLU(W0@x+b0), h1=ReLU(W1@h0+b1), z=w2@h1+b2, f(x)=1[z>=0]"
                    if dense_gauge
                    else "2-layer ReLU net: h1=ReLU(W1@x+b1), z=w2@h1+b2, f(x)=1[z>=0]"
                )
                if allow_python:
                    wb_note = (
                        "Access mode: WHITE-BOX (ReLU weights) + ORACLE. The exact neural network weights "
                        f"implementing f(x) are saved in your workspace as `weights.npz` ({arch_desc}). "
                        "You may analyze `weights.npz` using `run_python` (numpy/scipy available) and/or query the oracle.\n"
                    )
                else:
                    wb_note = (
                        f"Access mode: WHITE-BOX (ReLU weight summary, no Python tool) + ORACLE ({arch_desc}).\n"
                        f"Shapes: W1={bundle['W1'].shape}, b1={bundle['b1'].shape}, w2={bundle['w2'].shape}, "
                        f"b2={float(bundle['b2']):.4f}, ||W1^T w2||_2={float(np.linalg.norm(inst.W1.T @ inst.w2)):.3e}.\n"
                    )
            user_prompt = (
                base_prompt
                + wb_note
                + 'Return your final answer as JSON: {"decision": "PLANTED" | "CLEAN", "confidence": 0.5, "rationale": "..."}'
            )

        res = run_agent_loop(
            client=client,
            model=model,
            system=system,
            user_prompt=user_prompt,
            oracle=oracle,
            sandbox=sbx if allow_python else None,
            allow_python=allow_python,
            max_turns=max_turns,
            max_tokens_per_turn=max_tokens,
            thinking_budget=thinking_budget,
            thinking_mode=thinking_mode,
            effort=effort,
        )

    dec_int, dec_lbl, parse_ok = parse_exp1_decision(res["final_text"])
    correct = int(dec_int == planted) if dec_int is not None else 0
    return {
        "experiment": "exp1_vbb",
        "model": model,
        "thinking_budget": thinking_budget,
        "thinking_mode": thinking_mode,
        "effort": effort or "default",
        "family": family,
        "regime": regime,
        "substrate": substrate,
        "mode": mode,
        "track": track,
        "seed": seed,
        "instance_id": inst.instance_id,
        "n": n,
        "m": m,
        "k": k,
        "eta": eta,
        "planted": planted,
        "decision_int": dec_int,
        "decision_label": dec_lbl,
        "parse_ok": parse_ok,
        "correct": correct,
        **{
            k_: res[k_]
            for k_ in (
                "turns",
                "input_tokens",
                "output_tokens",
                "thinking_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "truncated_turns",
                "stop_reasons",
                "cost_usd",
                "queries_used",
                "python_calls",
                "wall_s",
                "tool_sequence",
                "final_excerpt",
            )
        },
    }


# ==============================================================================
# EXPERIMENT 2: C_inv COMPUTATIONAL BLACKWELL INVERSION (BUDGET-PHASE CURVE)
# ==============================================================================
def run_exp2_trial(
    *,
    client: Any,
    model: str,
    thinking_budget: int = 0,
    thinking_mode: str = "adaptive",
    effort: str = "medium",
    variant: str,
    mode: str,
    budget_track: str,
    seed: int,
    n: int = 40,
    support_size: int = 3,
    q_max: int = 80,
    task_type: str = "majority3",
    max_tokens_override: int = 0,
) -> Dict[str, Any]:
    inst = generate_blackwell_inv_instance(
        n=n, support_size=support_size, variant=variant, seed=seed, task_type=task_type
    )

    # Budget tracks mapping the phase curve Delta_WB(B):
    #   - matched_total_budget    : tight equal budget (3t BB / 3t WB)
    #   - matched_medium_budget   : medium equal budget (5t BB / 5t WB)
    #   - matched_query_free_code : asymmetric wide WB (4t BB / 8t WB)
    #   - matched_wide_equal_budget: wide equal budget (8t BB / 8t WB) [B10 control]
    tok_cap = int(max_tokens_override) if max_tokens_override > 0 else (8192 if effort in ("high", "xhigh", "max") else 4096)
    if budget_track == "matched_total_budget":
        max_turns = 3
        max_tokens = tok_cap
    elif budget_track == "matched_medium_budget":
        max_turns = 5
        max_tokens = tok_cap
    elif budget_track == "matched_query_free_code":
        max_turns = 8 if mode == "WB" else 4
        max_tokens = tok_cap
    elif budget_track == "matched_wide_equal_budget":
        max_turns = 8
        max_tokens = tok_cap
    else:
        raise ValueError(f"Unknown budget_track={budget_track}")

    if task_type == "dtree2":
        system = (
            "You are an algorithmic reverse-engineering and statistical learning specialist. "
            "Determine the exact depth-2 decision tree governing f(x) on uniform inputs "
            "x in {-1, +1}^40, and output strictly valid JSON: "
            '{"support": [root, leaf_pos, leaf_neg], "signs": [s_root, s_pos, s_neg]}.'
        )
        base_prompt = (
            f"Task: A deterministic binary policy function f: {{-1, +1}}^{n} -> {{0, 1}} behaves on "
            "uniform random inputs as a depth-2 decision tree over 3 distinct coordinates:\n"
            "    if s_root * x[root] > 0: return 1 if s_pos * x[leaf_pos] > 0 else 0\n"
            "    else:                    return 1 if s_neg * x[leaf_neg] > 0 else 0\n"
            f"where root, leaf_pos, leaf_neg are distinct indices in 0..{n-1} and each sign is in {{-1, +1}}.\n"
            f"You have black-box oracle tools (`query_oracle_random` and `query_oracle`, total budget {q_max} queries) "
            "and a Python 3 sandbox (`run_python`, with numpy and scipy; all oracle responses are stored in `queries.json`).\n"
        )
        json_hint = 'Conclude with JSON: {"support": [root, leaf_pos, leaf_neg], "signs": [s_root, s_pos, s_neg]}.'
    elif task_type == "dlist3":
        system = (
            "You are an algorithmic reverse-engineering and statistical learning specialist. "
            "Determine the exact depth-3 decision list governing f(x) on uniform inputs "
            "x in {-1, +1}^40, and output strictly valid JSON: "
            '{"support": [i1, i2, i3], "signs": [s1, s2, s3]}.'
        )
        base_prompt = (
            f"Task: A deterministic binary policy function f: {{-1, +1}}^{n} -> {{0, 1}} behaves on "
            "uniform random inputs as a depth-3 decision list over 3 distinct ordered coordinates:\n"
            "    if s_1 * x[i_1] > 0: return 1\n"
            "    if s_2 * x[i_2] > 0: return 0\n"
            "    return 1 if s_3 * x[i_3] > 0 else 0\n"
            f"where i_1, i_2, i_3 are distinct indices in 0..{n-1} and each s_j is in {{-1, +1}}.\n"
            f"You have black-box oracle tools (`query_oracle_random` and `query_oracle`, total budget {q_max} queries) "
            "and a Python 3 sandbox (`run_python`, with numpy and scipy; all oracle responses are stored in `queries.json`).\n"
        )
        json_hint = 'Conclude with JSON: {"support": [i1, i2, i3], "signs": [s1, s2, s3]}.'
    else:
        system = (
            "You are an algorithmic reverse-engineering and statistical learning specialist. "
            "Determine the exact 3-variable majority decision rule governing f(x) on uniform inputs "
            "x in {-1, +1}^40, and output strictly valid JSON: "
            '{"support": [i1, i2, i3], "signs": [s1, s2, s3]}.'
        )
        base_prompt = (
            f"Task: A deterministic binary policy function f: {{-1, +1}}^{n} -> {{0, 1}} behaves on "
            f"uniform random inputs as a {support_size}-variable majority rule:\n"
            "    f(x) = 1 if (s_1 * x[i_1] + s_2 * x[i_2] + s_3 * x[i_3]) > 0 else 0,\n"
            f"where i_1 < i_2 < i_3 are distinct indices in 0..{n-1} and each s_j is in {{-1, +1}}.\n"
            f"You have black-box oracle tools (`query_oracle_random` and `query_oracle`, total budget {q_max} queries) "
            "and a Python 3 sandbox (`run_python`, with numpy and scipy; all oracle responses are stored in `queries.json`).\n"
        )
        json_hint = 'Conclude with JSON: {"support": [i1, i2, i3], "signs": [s1, s2, s3]}.'

    with IsolatedPythonSandbox(timeout_s=10.0) as sbx:
        oracle = MeteredOracle(inst, max_queries=q_max, queries_json_path=sbx.queries_json_path)

        if mode == "BB":
            wb_note = "Access mode: BLACK-BOX ONLY (oracle + Python workspace).\n"
        else:
            if variant in ("wb_clean", "wb_decoy_trivial_guard", "wb_decoy_long_refutable_guard", "wb_inv_code"):
                code_str = inst.export_whitebox_code()
                sbx.write_text_artifact("artifact.py", code_str)
                wb_note = (
                    "Access mode: WHITE-BOX + ORACLE. In addition to the oracle tools, the exact "
                    "white-box Python implementation of f(x) is available in your workspace as "
                    "`artifact.py`. You may inspect it, query the oracle, or combine both however you judge best.\n"
                )
            elif variant == "wb_inv_relu":
                bundle = inst.export_whitebox_relu_bundle()
                sbx.write_npz_artifact("weights.npz", bundle)
                wb_note = (
                    "Access mode: WHITE-BOX (ReLU weights) + ORACLE. In addition to the oracle tools, "
                    "the exact 3-hidden-layer ReLU neural network weights implementing f(x) are saved "
                    "in your workspace as `weights.npz` (arrays W1, b1, W2, b2, W3, b3, w4, b4; "
                    "h1=ReLU(W1@x+b1), h2=ReLU(W2@h1+b2), h3=ReLU(W3@h2+b3), f(x)=1[w4@h3+b4 >= 0]). "
                    "You may inspect `weights.npz`, query the oracle, or combine both however you judge best.\n"
                )
            else:
                raise ValueError(f"Unknown variant={variant}")

        user_prompt = base_prompt + wb_note + json_hint

        res = run_agent_loop(
            client=client,
            model=model,
            system=system,
            user_prompt=user_prompt,
            oracle=oracle,
            sandbox=sbx,
            allow_python=True,
            max_turns=max_turns,
            max_tokens_per_turn=max_tokens,
            thinking_budget=thinking_budget,
            thinking_mode=thinking_mode,
            effort=effort,
        )

    pred_supp, pred_sgns, parse_ok = parse_exp2_rule(res["final_text"], n=n, support_size=support_size)
    metrics = score_rule_prediction(inst, pred_supp, pred_sgns)
    return {
        "experiment": "exp2_blackwell",
        "model": model,
        "thinking_budget": thinking_budget,
        "thinking_mode": thinking_mode,
        "effort": effort or "default",
        "task_type": task_type,
        "variant": variant,
        "mode": mode,
        "budget_track": budget_track,
        "seed": seed,
        "instance_id": inst.instance_id,
        "n": n,
        "q_max": q_max,
        "true_support": list(inst.true_support),
        "true_signs": list(inst.true_signs),
        "decoy_support": list(inst.decoy_support),
        "pred_support": pred_supp,
        "pred_signs": pred_sgns,
        "parse_ok": parse_ok,
        **metrics,
        **{
            k_: res[k_]
            for k_ in (
                "turns",
                "input_tokens",
                "output_tokens",
                "thinking_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "truncated_turns",
                "stop_reasons",
                "cost_usd",
                "queries_used",
                "python_calls",
                "wall_s",
                "tool_sequence",
                "final_excerpt",
            )
        },
    }


# ==============================================================================
# RESUMABLE JSONL DRIVER & CLI
# ==============================================================================
def load_completed_keys(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "run_key" in rec:
                    done.add(str(rec["run_key"]))
            except json.JSONDecodeError:
                continue
    return done


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def build_exp1_schedule(
    models: Sequence[str],
    family: str,
    regimes: Sequence[str],
    substrates: Sequence[str],
    tracks: Sequence[str],
    n_pairs: int,
    seed_offset: int = 1000,
    thinking_mode: str = "adaptive",
    effort: str = "medium",
    skip_bb: bool = False,
) -> List[Dict[str, Any]]:
    """Build balanced P/Q schedule (n_pairs planted + n_pairs clean = 2*n_pairs per cell)."""
    jobs: List[Dict[str, Any]] = []
    ef_tag = f"|tm={thinking_mode}|ef={effort or 'default'}"
    for model in models:
        for track in tracks:
            for regime in regimes:
                # One shared BB reference per (model, track, regime) + WB per substrate
                mode_subs = ([] if skip_bb else [("BB", "blackbox")]) + [("WB", s) for s in substrates]
                for mode, sub in mode_subs:
                    for i in range(n_pairs):
                        for planted in (0, 1):
                            seed = seed_offset + 100 * i + planted
                            rkey = f"exp1|{model}{ef_tag}|{family}|{track}|{regime}|{mode}|{sub}|s{seed}|b{planted}"
                            jobs.append(
                                {
                                    "run_key": rkey,
                                    "model": model,
                                    "family": family,
                                    "track": track,
                                    "regime": regime,
                                    "mode": mode,
                                    "substrate": sub,
                                    "seed": seed,
                                    "planted": planted,
                                }
                            )
    return jobs


def build_exp2_schedule(
    models: Sequence[str],
    variants: Sequence[str],
    budget_tracks: Sequence[str],
    n_seeds: int,
    seed_offset: int = 5000,
    thinking_mode: str = "adaptive",
    effort: str = "medium",
    task_type: str = "majority3",
    skip_bb: bool = False,
    only_bb: bool = False,
    max_tokens_override: int = 0,
) -> List[Dict[str, Any]]:
    """Build matched-pair BB vs WB schedule (identical seeds across BB and all WB variants)."""
    jobs: List[Dict[str, Any]] = []
    ef_tag = f"|tm={thinking_mode}|ef={effort or 'default'}"
    tt_tag = "" if task_type == "majority3" else f"|tt={task_type}"
    mt_tag = "" if max_tokens_override <= 0 else f"|mt={max_tokens_override}"
    for model in models:
        for track in budget_tracks:
            for i in range(n_seeds):
                seed = seed_offset + i
                if not skip_bb:
                    # Paired BB baseline (variant tag 'wb_inv_code' shares identical f_true for same seed!)
                    rkey_bb = f"exp2|{model}{ef_tag}{tt_tag}{mt_tag}|{track}|BB|blackbox|s{seed}"
                    jobs.append(
                        {
                            "run_key": rkey_bb,
                            "model": model,
                            "budget_track": track,
                            "mode": "BB",
                            "variant": "wb_inv_code",
                            "seed": seed,
                            "task_type": task_type,
                        }
                    )
                if not only_bb:
                    for var in variants:
                        rkey_wb = f"exp2|{model}{ef_tag}{tt_tag}{mt_tag}|{track}|WB|{var}|s{seed}"
                        jobs.append(
                            {
                                "run_key": rkey_wb,
                                "model": model,
                                "budget_track": track,
                                "mode": "WB",
                                "variant": var,
                                "seed": seed,
                                "task_type": task_type,
                            }
                        )
    return jobs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pilot runner for Observer-Relative VBB & Computational Blackwell Inversion."
    )
    parser.add_argument("--list-models", action="store_true", help="List available Anthropic models and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Run full tool/sandbox pipeline with $0 mock client.")
    parser.add_argument(
        "--experiment",
        choices=("exp1", "exp2", "all"),
        default="all",
        help="Which experiment suite to execute.",
    )
    parser.add_argument(
        "--models",
        default="claude-haiku-5-5,claude-sonnet-5-5",
        help="Comma-separated Anthropic model IDs.",
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("adaptive", "default", "disabled", "budget"),
        default="adaptive",
        help="Anthropic thinking mode (adaptive works across Claude 4.6+ and 5.5).",
    )
    parser.add_argument(
        "--effort",
        choices=("", "low", "medium", "high", "xhigh", "max"),
        default="medium",
        help="Anthropic output_config.effort level.",
    )
    parser.add_argument("--thinking-budget", type=int, default=0, help="Legacy budget_tokens (used with --thinking-mode budget).")
    parser.add_argument("--family", choices=("4xor", "4xor_strict", "4xor_n40", "3xor", "4xor_fixed120"), default="4xor_strict")
    parser.add_argument("--task-type", choices=("majority3", "dtree2", "dlist3"), default="majority3")
    parser.add_argument(
        "--regimes",
        default="control_impossible,vbb_hard,transition,transparent",
        help="Comma-separated Exp 1 regimes.",
    )
    parser.add_argument(
        "--substrates",
        default="code_clean,code_obf,relu_weights",
        help="Comma-separated Exp 1 WB substrates.",
    )
    parser.add_argument("--tracks", default="tool_python", help="Comma-separated Exp 1 tracks (tool_free,tool_python).")
    parser.add_argument("--exp1-pairs", type=int, default=20, help="Number of (P, Q) pairs per Exp 1 cell (total 2*pairs).")
    parser.add_argument("--exp1-seed-offset", type=int, default=1000, help="Seed offset for Exp 1 instances.")
    parser.add_argument("--exp1-skip-bb", action="store_true", help="Skip BB baseline in Exp 1 schedule.")
    parser.add_argument(
        "--variants",
        default="wb_clean,wb_inv_code,wb_inv_relu",
        help="Comma-separated Exp 2 WB variants.",
    )
    parser.add_argument(
        "--budget-tracks",
        default="matched_total_budget,matched_query_free_code",
        help="Comma-separated Exp 2 budget tracks.",
    )
    parser.add_argument("--exp2-seeds", type=int, default=30, help="Number of matched seeds per Exp 2 cell.")
    parser.add_argument("--skip-bb", action="store_true", help="Skip BB baseline in Exp 2 schedule.")
    parser.add_argument("--only-bb", action="store_true", help="Run ONLY BB baseline in Exp 2 schedule.")
    parser.add_argument("--max-tokens-override", type=int, default=0, help="Override max_tokens_per_turn for Exp 2 (e.g. 8192).")
    parser.add_argument("--resume-from", default="", help="Comma-separated extra read-only JSONL files to skip already-completed keys.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent worker threads for API calls.")
    parser.add_argument("--min-interval-s", type=float, default=0.0, help="Min seconds between Gemini API calls (for tight TPM models like Gemma 4).")
    parser.add_argument("--max-cost-usd", type=float, default=35.0, help="Hard API spend cap in USD.")
    parser.add_argument("--out", default="pilot_results.jsonl", help="Output JSONL log path (resumable).")

    args = parser.parse_args(argv)

    if args.list_models:
        client = AnthropicClient(max_cost_usd=0.0)
        models = client.list_models()
        print(f"Found {len(models)} Anthropic models:")
        for m in models:
            print(f"  - {m.get('id')}  (display={m.get('display_name', '')}, created={m.get('created_at', '')})")
        return 0

    models_list = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.dry_run:
        models_list = [f"mock-{m}" if not m.startswith("mock-") else m for m in models_list]
        client: Any = MockDryRunClient(max_cost_usd=0.0)
    elif all(m.lower().startswith(("gemini-", "gemma-")) for m in models_list):
        client = GeminiClient(max_cost_usd=args.max_cost_usd, min_interval_s=args.min_interval_s)
    else:
        client = AnthropicClient(max_cost_usd=args.max_cost_usd)

    out_path = Path(args.out)
    done = load_completed_keys(out_path)
    if args.resume_from:
        for extra_p in args.resume_from.split(","):
            extra_p = extra_p.strip()
            if extra_p:
                done |= load_completed_keys(Path(extra_p))
    executed = 0
    skipped = 0
    errors = 0
    write_lock = threading.Lock()
    stop_event = threading.Event()

    pending_jobs: List[Tuple[str, Dict[str, Any]]] = []
    if args.experiment in ("exp1", "all"):
        regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
        substrates = [s.strip() for s in args.substrates.split(",") if s.strip()]
        tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
        for job in build_exp1_schedule(
            models_list,
            args.family,
            regimes,
            substrates,
            tracks,
            args.exp1_pairs,
            seed_offset=args.exp1_seed_offset,
            thinking_mode=args.thinking_mode,
            effort=args.effort,
            skip_bb=args.exp1_skip_bb,
        ):
            if job["run_key"] in done:
                skipped += 1
            else:
                pending_jobs.append(("exp1", job))

    if args.experiment in ("exp2", "all"):
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        btracks = [b.strip() for b in args.budget_tracks.split(",") if b.strip()]
        for job in build_exp2_schedule(
            models_list,
            variants,
            btracks,
            args.exp2_seeds,
            thinking_mode=args.thinking_mode,
            effort=args.effort,
            task_type=args.task_type,
            skip_bb=args.skip_bb,
            only_bb=args.only_bb,
            max_tokens_override=args.max_tokens_override,
        ):
            if job["run_key"] in done:
                skipped += 1
            else:
                pending_jobs.append(("exp2", job))

    def _run_one(kind: str, job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if stop_event.is_set():
            return None
        if kind == "exp1":
            rec = run_exp1_trial(
                client=client,
                model=job["model"],
                thinking_budget=args.thinking_budget,
                thinking_mode=args.thinking_mode,
                effort=args.effort,
                family=job["family"],
                regime=job["regime"],
                substrate=job["substrate"],
                mode=job["mode"],
                track=job["track"],
                seed=job["seed"],
                planted=job["planted"],
            )
        else:
            rec = run_exp2_trial(
                client=client,
                model=job["model"],
                thinking_budget=args.thinking_budget,
                thinking_mode=args.thinking_mode,
                effort=args.effort,
                variant=job["variant"],
                mode=job["mode"],
                budget_track=job["budget_track"],
                seed=job["seed"],
                task_type=job.get("task_type", "majority3"),
                max_tokens_override=args.max_tokens_override,
            )
        if args.max_tokens_override > 0:
            rec["max_tokens_override"] = int(args.max_tokens_override)
        rec["run_key"] = job["run_key"]
        return rec

    total_pending = len(pending_jobs)
    workers = max(1, int(args.workers))
    if workers == 1:
        try:
            for kind, job in pending_jobs:
                rec = _run_one(kind, job)
                if rec is not None:
                    append_jsonl(out_path, rec)
                    done.add(job["run_key"])
                    executed += 1
        except BudgetExceededError as exc:
            print(f"[STOPPED — BUDGET CAP] {exc}", file=sys.stderr)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_run_one, kind, job): job["run_key"] for kind, job in pending_jobs}
            for fut in concurrent.futures.as_completed(futs):
                rkey = futs[fut]
                try:
                    rec = fut.result()
                except BudgetExceededError as exc:
                    if not stop_event.is_set():
                        stop_event.set()
                        print(f"[STOPPED — BUDGET CAP] {exc}", file=sys.stderr)
                    continue
                except Exception as exc:
                    errors += 1
                    print(f"[TRIAL ERROR] {rkey}: {exc}", file=sys.stderr)
                    continue
                if rec is not None:
                    with write_lock:
                        append_jsonl(out_path, rec)
                        done.add(rkey)
                        executed += 1
                        if executed % 25 == 0 or executed == total_pending:
                            print(
                                f"  progress: {executed}/{total_pending} "
                                f"(errors={errors}, cost=${client.total_cost_usd:.4f})",
                                flush=True,
                            )

    print(
        f"Done. executed={executed}, skipped_resumed={skipped}, errors={errors}, "
        f"total_cost_usd=${client.total_cost_usd:.4f}, out={out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
