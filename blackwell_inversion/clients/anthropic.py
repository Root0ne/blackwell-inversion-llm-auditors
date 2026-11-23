"""Standard-library client for the Anthropic Messages API (version 2023-06-01)."""

from __future__ import annotations

import copy
import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .base import COST_KEY, Message, ModelClient, ToolSchema, http_error_body, resolve_api_key
from .pricing import estimate_cost_usd

USER_AGENT = "vbb-blackwell-pilot/1.0"
RETRYABLE_HTTP_CODES = (429, 500, 502, 503, 529)
MAX_RETRY_SLEEP_S = 30.0
MAX_NETWORK_RETRY_SLEEP_S = 20.0

# Older models that accept temperature=0.0; it is sent only when thinking is not configured.
ZERO_TEMPERATURE_MODEL_PREFIXES = (
    "claude-3-",
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
)

EPHEMERAL_CACHE = {"type": "ephemeral"}


def apply_prompt_caching(
    system: str,
    tools: Optional[List[ToolSchema]],
    messages: List[Message],
) -> Tuple[Any, Optional[List[ToolSchema]], List[Message]]:
    """Return deep copies of the payload parts with ephemeral prompt-cache breakpoints.

    Breakpoints go on the system prompt, the last tool and, from the third message on,
    the latest user turn, so the history prefix is cached across turns. The model sees
    the same input; only billing changes.
    """
    sys_cached: Any = [{"type": "text", "text": system, "cache_control": dict(EPHEMERAL_CACHE)}]
    tools_cached = copy.deepcopy(tools) if tools else None
    if tools_cached:
        tools_cached[-1]["cache_control"] = dict(EPHEMERAL_CACHE)
    msgs_cached = copy.deepcopy(messages)
    if len(msgs_cached) >= 3:
        for m in reversed(msgs_cached):
            if m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str):
                    m["content"] = [{"type": "text", "text": content, "cache_control": dict(EPHEMERAL_CACHE)}]
                elif isinstance(content, list) and content:
                    content[-1]["cache_control"] = dict(EPHEMERAL_CACHE)
                break
    return sys_cached, tools_cached, msgs_cached


def thinking_config(thinking_mode: str, thinking_budget: int) -> Optional[Dict[str, Any]]:
    """The `thinking` request field for a CLI thinking mode, or None to omit it.

    Modes: "adaptive" and "disabled" are sent as such; "budget" (and "default" with a
    positive budget) sends an explicit token budget; "default" without one omits the field.
    """
    if thinking_mode == "adaptive":
        return {"type": "adaptive"}
    if thinking_mode == "disabled":
        return {"type": "disabled"}
    if thinking_budget > 0:
        return {"type": "enabled", "budget_tokens": int(thinking_budget)}
    return None


class AnthropicClient(ModelClient):
    """Anthropic Messages API client with retries, prompt caching and the spend cap."""

    BASE_URL = "https://api.anthropic.com/v1"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_cost_usd: float = 40.0,
        timeout_s: float = 120.0,
        max_retries: int = 5,
    ) -> None:
        super().__init__(max_cost_usd)
        self.api_key = resolve_api_key(api_key, "ANTHROPIC_API_KEY")
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
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
            "user-agent": USER_AGENT,
        }

    def list_models(self) -> List[Dict[str, Any]]:
        """Return the model descriptors visible to this API key (GET /v1/models)."""
        req = urllib.request.Request(
            f"{self.BASE_URL}/models?limit=100",
            headers=self._headers(),
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return list(payload.get("data", []))

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
        """The JSON body of one POST /v1/messages request."""
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

        thinking = thinking_config(thinking_mode, thinking_budget)
        if thinking is not None:
            if thinking["type"] == "enabled" and body["max_tokens"] <= thinking_budget:
                body["max_tokens"] = thinking_budget + 1024
            body["thinking"] = thinking

        if effort:
            body["output_config"] = {"effort": effort}

        if "thinking" not in body and model.startswith(ZERO_TEMPERATURE_MODEL_PREFIXES):
            body["temperature"] = 0.0
        return body

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
                return self._account(model, res)
            except urllib.error.HTTPError as err:
                last_err = err
                err_body = http_error_body(err)
                if err.code in RETRYABLE_HTTP_CODES and attempt < self.max_retries - 1:
                    retry_after = err.headers.get("retry-after") if err.headers else None
                    sleep_s = float(retry_after) if retry_after and retry_after.isdigit() else (2.0**attempt)
                    time.sleep(min(sleep_s, MAX_RETRY_SLEEP_S))
                    continue
                raise RuntimeError(f"Anthropic HTTP {err.code}: {err_body}") from err
            except urllib.error.URLError as err:
                last_err = err
                if attempt < self.max_retries - 1:
                    time.sleep(min(2.0**attempt, MAX_NETWORK_RETRY_SLEEP_S))
                    continue
                raise RuntimeError(f"Anthropic network error: {err}") from err
        raise RuntimeError(f"Anthropic API failed after retries: {last_err}")

    def _account(self, model: str, res: Dict[str, Any]) -> Dict[str, Any]:
        """Price the call, add it to the totals and attach the cost to the response."""
        usage = res.get("usage", {})
        in_tok = int(usage.get("input_tokens", 0))
        out_tok = int(usage.get("output_tokens", 0))
        cw_tok = int(usage.get("cache_creation_input_tokens", 0))
        cr_tok = int(usage.get("cache_read_input_tokens", 0))
        cost = estimate_cost_usd(model, in_tok, out_tok, cache_write_tokens=cw_tok, cache_read_tokens=cr_tok)
        self.record_usage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_write_tokens=cw_tok,
            cache_read_tokens=cr_tok,
            cost_usd=cost,
        )
        res[COST_KEY] = cost
        return res

