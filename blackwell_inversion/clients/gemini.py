"""Standard-library client for the Google Gemini Developer API (v1beta generateContent).

Requests arrive in Anthropic Messages form and are translated to Gemini `contents`;
replies are translated back, so the agent loop is provider-agnostic. The model's raw
parts, including opaque `thoughtSignature` fields, are kept in a private content block
and replayed verbatim on the next turn, as the API requires for multi-turn tool use.
Gemma models take no system instruction, so the system prompt is inlined into the
first user turn for them.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .base import COST_KEY, BudgetExceededError, Message, ModelClient, ToolSchema, http_error_body, resolve_api_key
from .pricing import estimate_cost_usd

USER_AGENT = "vbb-blackwell-pilot/1.0"
RAW_PARTS_BLOCK = "_gemini_raw_parts"
RETRYABLE_HTTP_CODES = (429, 500, 502, 503, 504)
MAX_NETWORK_RETRY_SLEEP_S = 20.0

_SCHEMA_TYPES = {
    "object": "OBJECT",
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
}


def to_gemini_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively translate a JSON Schema tool `input_schema` to the Gemini OpenAPI subset."""
    out: Dict[str, Any] = {}
    if "type" in schema:
        t = str(schema["type"]).lower()
        out["type"] = _SCHEMA_TYPES.get(t, t.upper())
    if "description" in schema:
        out["description"] = str(schema["description"])
    if "enum" in schema and out.get("type") == "STRING":
        out["enum"] = [str(v) for v in schema["enum"]]
    if "properties" in schema and isinstance(schema["properties"], dict):
        out["properties"] = {k: to_gemini_schema(v) for k, v in schema["properties"].items()}
    if "items" in schema and isinstance(schema["items"], dict):
        out["items"] = to_gemini_schema(schema["items"])
    if "required" in schema and isinstance(schema["required"], list):
        out["required"] = list(schema["required"])
    return out


def _assistant_parts(content: Any) -> List[Dict[str, Any]]:
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
                parts.append({"functionCall": {"name": str(blk.get("name", "")), "args": blk.get("input", {}) or {}}})
    return parts


def _tool_result_part(blk: Dict[str, Any], id_to_name: Dict[str, str]) -> Dict[str, Any]:
    tname = id_to_name.get(str(blk.get("tool_use_id", "")), "tool")
    raw_res = str(blk.get("content", ""))
    try:
        parsed_res = json.loads(raw_res)
        if not isinstance(parsed_res, dict):
            parsed_res = {"result": parsed_res}
    except json.JSONDecodeError:
        parsed_res = {"result": raw_res}
    return {"functionResponse": {"name": tname, "response": parsed_res}}


def convert_messages(messages: List[Message], inline_system: str = "") -> List[Dict[str, Any]]:
    """Translate Anthropic-form messages to Gemini `contents`.

    Assistant turns that carry a raw-parts block are replayed verbatim; tool results are
    matched to their call by id to recover the function name. `inline_system`, when
    given, is prepended to the first user turn.
    """
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
                    if isinstance(blk, dict) and blk.get("type") == RAW_PARTS_BLOCK:
                        raw_parts = blk.get("parts")
            parts = raw_parts if raw_parts is not None else _assistant_parts(content)
            if raw_parts is not None or parts:
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
                        parts.append(_tool_result_part(blk, id_to_name))
            if parts:
                contents.append({"role": "user", "parts": parts})
    return contents


def _is_gemma(model: str) -> bool:
    return model.lower().startswith("gemma-")


def generation_config(model: str, max_tokens: int, thinking_mode: str, thinking_budget: int, effort: str) -> Dict[str, Any]:
    """The `generationConfig` field: Gemini 3/4 take a thinking level, Gemini 2.5 a token budget."""
    gen_cfg: Dict[str, Any] = {"maxOutputTokens": int(max_tokens)}
    if _is_gemma(model):
        return gen_cfg
    if model.lower().startswith(("gemini-3", "gemini-4")):
        if thinking_mode == "disabled":
            gen_cfg["thinkingConfig"] = {"thinkingLevel": "minimal"}
        elif effort in ("low", "medium", "high"):
            gen_cfg["thinkingConfig"] = {"thinkingLevel": effort}
    elif model.lower().startswith("gemini-2.5") and thinking_budget > 0:
        gen_cfg["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}
    return gen_cfg


class GeminiClient(ModelClient):
    """Gemini/Gemma client with retries, optional request pacing and the spend cap."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_cost_usd: float = 25.0,
        timeout_s: float = 150.0,
        max_retries: int = 6,
        min_interval_s: float = 0.0,
    ) -> None:
        super().__init__(max_cost_usd)
        self.api_key = resolve_api_key(api_key, "GEMINI_API_KEY")
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.min_interval_s = float(min_interval_s)
        self._last_call_ts = 0.0

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
        return {
            "x-goog-api-key": self.api_key,
            "content-type": "application/json",
            "user-agent": USER_AGENT,
        }

    def build_request_body(
        self,
        *,
        model: str,
        system: str,
        messages: List[Message],
        max_tokens: int,
        tools: Optional[List[ToolSchema]],
        tool_choice: Optional[Dict[str, Any]],
        thinking_budget: int,
        thinking_mode: str,
        effort: str,
    ) -> Dict[str, Any]:
        """The JSON body of one generateContent request."""
        is_gemma = _is_gemma(model)
        body: Dict[str, Any] = {
            "contents": convert_messages(messages, inline_system=(system if is_gemma else "")),
            "generationConfig": generation_config(model, max_tokens, thinking_mode, thinking_budget, effort),
        }
        if system and not is_gemma:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            decls = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": to_gemini_schema(t.get("input_schema", {"type": "object"})),
                }
                for t in tools
            ]
            body["tools"] = [{"functionDeclarations": decls}]
            if tool_choice and tool_choice.get("type") == "none":
                body["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
        return body

    def _pace(self) -> None:
        """Keep at least min_interval_s between requests (for low tokens-per-minute quotas)."""
        if self.min_interval_s <= 0.0:
            return
        with self._lock:
            wait_s = self.min_interval_s - (time.monotonic() - self._last_call_ts)
            if wait_s > 0.0:
                time.sleep(wait_s)
            self._last_call_ts = time.monotonic()

    def create_message(
        self,
        *,
        model: str,
        system: str,
        messages: List[Message],
        max_tokens: int,
        tools: Optional[List[ToolSchema]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        thinking_budget: int = 0,
        thinking_mode: str = "adaptive",
        effort: str = "",
    ) -> Dict[str, Any]:
        self.check_budget()
        body = self.build_request_body(
            model=model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            thinking_budget=thinking_budget,
            thinking_mode=thinking_mode,
            effort=effort,
        )
        url = f"{self.BASE_URL}/models/{model}:generateContent"
        data = json.dumps(body).encode("utf-8")
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            self._pace()
            req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                return self._normalise(model, raw, n_messages=len(messages))
            except urllib.error.HTTPError as err:
                last_err = err
                err_body = http_error_body(err)
                if err.code == 402:
                    raise BudgetExceededError(f"Gemini HTTP 402 (credits depleted): {err_body}") from err
                if err.code in RETRYABLE_HTTP_CODES and attempt < self.max_retries - 1:
                    if err.code == 429 and _is_gemma(model):
                        sleep_s = min(6.0 * (2.0**attempt), 65.0)
                    else:
                        sleep_s = min(3.0 * (2.0**attempt), 45.0)
                    time.sleep(sleep_s)
                    continue
                raise RuntimeError(f"Gemini HTTP {err.code}: {err_body}") from err
            except (urllib.error.URLError, TimeoutError, OSError) as err:
                last_err = err
                if attempt < self.max_retries - 1:
                    time.sleep(min(2.0**attempt, MAX_NETWORK_RETRY_SLEEP_S))
                    continue
                raise RuntimeError(f"Gemini network/timeout error: {err}") from err
        raise RuntimeError(f"Gemini API failed after retries: {last_err}")

    def _normalise(self, model: str, raw: Dict[str, Any], n_messages: int) -> Dict[str, Any]:
        """Price the call, update the totals and convert the reply to the contract's response shape.

        Thinking tokens are billed as output. Thought parts are dropped from the visible
        text; tool-call ids are synthesised from the turn index and part position.
        """
        usage_m = raw.get("usageMetadata", {}) or {}
        prompt_tok = int(usage_m.get("promptTokenCount", 0))
        cached_tok = int(usage_m.get("cachedContentTokenCount", 0))
        uncached_in = max(0, prompt_tok - cached_tok)
        cand_tok = int(usage_m.get("candidatesTokenCount", 0))
        think_tok = int(usage_m.get("thoughtsTokenCount", 0))
        out_billed = cand_tok + think_tok

        cost = estimate_cost_usd(model, uncached_in, out_billed, cache_write_tokens=0, cache_read_tokens=cached_tok)
        self.record_usage(
            input_tokens=uncached_in,
            output_tokens=out_billed,
            cache_write_tokens=0,
            cache_read_tokens=cached_tok,
            cost_usd=cost,
        )

        cands = raw.get("candidates", []) or []
        cand0 = cands[0] if cands else {}
        finish = str(cand0.get("finishReason", "STOP")).upper()
        raw_parts = (cand0.get("content", {}) or {}).get("parts", []) or []

        norm_content: List[Dict[str, Any]] = [{"type": RAW_PARTS_BLOCK, "parts": raw_parts}]
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
                        "id": f"gcall_{n_messages}_{idx_p}",
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
            COST_KEY: cost,
        }

