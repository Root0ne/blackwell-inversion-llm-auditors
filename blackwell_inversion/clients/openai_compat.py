"""Standard-library client for OpenAI-compatible chat completion servers.

One request is `POST {base_url}/chat/completions`. Local inference servers and many
hosted APIs speak this format, so this client lets the benchmark run any model they
serve. Requests arrive in Anthropic Messages form (see `base`) and are translated both
ways, so the agent loop does not know which provider is in use:

    system prompt                       -> first message, role "system"
    assistant text + tool_use blocks    <-> "content" + "tool_calls" (function name,
                                            arguments as a JSON string)
    user tool_result blocks             -> one role "tool" message per result, with
                                            "tool_call_id"; text blocks of the same user
                                            turn follow as one role "user" message
    tool schemas                        -> {"type": "function", "function": {"name",
                                            "description", "parameters"}}
    tool_choice {"type": "none"}        -> "none" (otherwise the field is omitted)
    finish_reason tool_calls / length / stop
                                        -> stop_reason tool_use / max_tokens / end_turn

Request settings:

    reasoning_effort   sent only when --effort is non-empty (`--effort ""` omits it);
                       --thinking-mode and --thinking-budget are not sent.
    token cap          sent as "max_tokens" by default, or as "max_completion_tokens"
                       (`max_tokens_field`), for servers that accept only that name.
    temperature        not sent; the server default applies.

Replies from real servers vary. Each case below is decided deliberately and pinned by a
test in `tests/test_openai_compat.py`:

    missing "usage", or missing or non-numeric counts
        counted as 0 tokens and $0. The record then shows 0 tokens and the spend cap
        cannot see that call.
    "content": null, or a list of text parts
        null is no text; the "text" of each part of a list is joined.
    tool-call "arguments" that are empty
        read as {} (no arguments).
    tool-call "arguments" that are not valid JSON, or valid JSON but not an object
        kept as the raw string in the tool_use block's "input". The agent loop answers
        such a call with an error tool_result and charges no query budget; on the next
        request the raw string is sent back unchanged as the call's arguments.
    tool-call "arguments" already decoded to an object by the server
        used as they are.
    missing tool-call id, or an id repeated within one reply
        replaced by "call_<n>_<i>", where n is the number of messages in the request
        and i the call's position in the reply, so ids are stable and unique.
    tool calls with finish_reason "stop" (or none, or any value other than "length")
        stop_reason "tool_use": the calls are executed. With "length" the reply was cut
        at the token cap and stop_reason is "max_tokens", as with the other providers.
    reasoning text in "reasoning_content" or "reasoning"
        dropped: not part of the answer and not sent back, as thinking blocks and
        thought parts are with the other providers. Its token count is read from
        usage.completion_tokens_details.reasoning_tokens when the server reports it.
        Reasoning written inline in "content" (for example between think tags) stays
        in the text; the answer parsers read the last matching JSON object.
    tool calls on a turn sent with tool_choice "none"
        passed on unchanged; the agent loop executes them and ends the trial.

A reply that cannot be interpreted at all (not JSON, an "error" object, no "choices",
a choice without a "message" object, "tool_calls" that is not a list, a tool call
without a function name) raises RuntimeError; no record is written for that trial.

Errors: 429 and 5xx are retried with exponential backoff (Retry-After is honoured);
other HTTP errors are raised at once; 402 (payment required) raises
BudgetExceededError like the spend cap. Network errors and timeouts are retried.

Cost: prices of arbitrary models are unknown, so the caller gives USD prices per million
input and output tokens (default 0, a local model). The --max-cost-usd cap is enforced
as with the other clients. The API key is optional: it is read from the environment
variable named by `api_key_env` (or from ~/.env) and sent as a bearer token; it is never
included in an exception message.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from .base import COST_KEY, BudgetExceededError, Message, ModelClient, ToolSchema, http_error_body, resolve_api_key

USER_AGENT = "vbb-blackwell-pilot/1.0"
RETRYABLE_HTTP_CODES = (429, 500, 502, 503, 504)
MAX_RETRY_SLEEP_S = 30.0
MAX_NETWORK_RETRY_SLEEP_S = 20.0
MAX_TOKENS_FIELDS = ("max_tokens", "max_completion_tokens")
REDACTED = "[REDACTED]"


def to_openai_tools(tools: List[ToolSchema]) -> List[Dict[str, Any]]:
    """Tool schemas in function-calling form; the JSON Schema is passed unchanged."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object"}),
            },
        }
        for t in tools
    ]


def _block_text(content: Any) -> str:
    """Text of a tool_result content: a string, or the text blocks of a block list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(b.get("text", "")) for b in content if isinstance(b, dict) and b.get("type") == "text")
    return "" if content is None else str(content)


def _arguments_text(tool_input: Any) -> str:
    """Arguments of a tool call as sent back: raw text stays raw, an object is JSON-encoded."""
    return tool_input if isinstance(tool_input, str) else json.dumps(tool_input)


def convert_messages(system: str, messages: List[Message]) -> List[Dict[str, Any]]:
    """Translate the system prompt and Anthropic-form messages to chat completion messages."""
    out: List[Dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            if isinstance(content, str):
                out.append({"role": "assistant", "content": content})
                continue
            texts: List[str] = []
            calls: List[Dict[str, Any]] = []
            for blk in content if isinstance(content, list) else []:
                if not isinstance(blk, dict):
                    continue
                if blk.get("type") == "text":
                    texts.append(str(blk.get("text", "")))
                elif blk.get("type") == "tool_use":
                    calls.append(
                        {
                            "id": str(blk.get("id", "")),
                            "type": "function",
                            "function": {"name": str(blk.get("name", "")), "arguments": _arguments_text(blk.get("input", {}))},
                        }
                    )
            entry: Dict[str, Any] = {"role": "assistant", "content": ("\n".join(texts) if texts else (None if calls else ""))}
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
            continue
        if isinstance(content, str):
            out.append({"role": "user", "content": content})
            continue
        user_texts: List[str] = []
        for blk in content if isinstance(content, list) else []:
            if not isinstance(blk, dict):
                continue
            if blk.get("type") == "tool_result":
                out.append(
                    {"role": "tool", "tool_call_id": str(blk.get("tool_use_id", "")), "content": _block_text(blk.get("content", ""))}
                )
            elif blk.get("type") == "text":
                user_texts.append(str(blk.get("text", "")))
        if user_texts:
            out.append({"role": "user", "content": "\n\n".join(user_texts)})
    return out


def _count(value: Any) -> int:
    """A token count from a usage object; anything that is not a non-negative number counts as 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    raise RuntimeError(f"OpenAI-compatible reply has message content of type {type(content).__name__}.")


def _parse_arguments(raw: Any) -> Any:
    """Tool-call arguments as an object, or the raw string when they are not a JSON object."""
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    text = raw if isinstance(raw, str) else json.dumps(raw)
    if not text.strip():
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    return parsed if isinstance(parsed, dict) else text


def normalise_reply(raw: Any, n_messages: int) -> Dict[str, Any]:
    """Convert a chat completion reply to the contract's response shape (without cost).

    Raises RuntimeError when the reply cannot be interpreted; see the module docstring.
    """
    if not isinstance(raw, dict):
        raise RuntimeError("OpenAI-compatible reply is not a JSON object.")
    if raw.get("error"):
        raise RuntimeError(f"OpenAI-compatible server returned an error: {str(raw['error'])[:500]}")
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("OpenAI-compatible reply has no choices.")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("OpenAI-compatible reply has no message.")

    content: List[Dict[str, Any]] = []
    text = _content_text(message.get("content"))
    if text:
        content.append({"type": "text", "text": text})

    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        raise RuntimeError("OpenAI-compatible reply has tool_calls that are not a list.")
    seen_ids: set[str] = set()
    for idx, call in enumerate(tool_calls):
        fn = call.get("function") if isinstance(call, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"OpenAI-compatible reply has tool call #{idx} without a function name.")
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id or call_id in seen_ids:
            call_id = f"call_{n_messages}_{idx}"
        seen_ids.add(call_id)
        content.append({"type": "tool_use", "id": call_id, "name": name, "input": _parse_arguments(fn.get("arguments"))})

    finish = str(choice.get("finish_reason") or "").lower()
    if finish == "length":
        stop_reason = "max_tokens"
    elif tool_calls:
        stop_reason = "tool_use"
    else:
        stop_reason = "end_turn"

    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
    return {
        "content": content,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": _count(usage.get("prompt_tokens")),
            "output_tokens": _count(usage.get("completion_tokens")),
            "output_tokens_details": {"thinking_tokens": _count(details.get("reasoning_tokens"))},
        },
    }


class OpenAICompatClient(ModelClient):
    """Client for any server with an OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: Optional[str] = None,
        api_key_env: str = "",
        max_cost_usd: float = 35.0,
        input_price_per_mtok: float = 0.0,
        output_price_per_mtok: float = 0.0,
        max_tokens_field: str = "max_tokens",
        timeout_s: float = 600.0,
        max_retries: int = 5,
    ) -> None:
        super().__init__(max_cost_usd)
        parsed = urllib.parse.urlparse(base_url or "")
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"base_url must be an http(s) URL such as http://127.0.0.1:8000/v1, got {base_url!r}")
        if max_tokens_field not in MAX_TOKENS_FIELDS:
            raise ValueError(f"max_tokens_field must be one of {MAX_TOKENS_FIELDS}, got {max_tokens_field!r}")
        if input_price_per_mtok < 0 or output_price_per_mtok < 0:
            raise ValueError("token prices must not be negative")
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.api_key = resolve_api_key(api_key, api_key_env) if (api_key or api_key_env) else ""
        self.input_price_per_mtok = float(input_price_per_mtok)
        self.output_price_per_mtok = float(output_price_per_mtok)
        self.max_tokens_field = max_tokens_field
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)

    def _headers(self) -> Dict[str, str]:
        if self.api_key_env and not self.api_key:
            raise RuntimeError(f"{self.api_key_env} environment variable is not set (named by --api-key-env).")
        headers = {"content-type": "application/json", "user-agent": USER_AGENT}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    def _redact(self, text: str) -> str:
        return text.replace(self.api_key, REDACTED) if self.api_key else text

    def estimate_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """USD cost of one call at the configured per-million-token prices."""
        return (input_tokens * self.input_price_per_mtok + output_tokens * self.output_price_per_mtok) / 1_000_000.0

    def build_request_body(
        self,
        *,
        model: str,
        system: str,
        messages: List[Message],
        max_tokens: int,
        tools: Optional[List[ToolSchema]],
        tool_choice: Optional[Dict[str, Any]],
        effort: str,
    ) -> Dict[str, Any]:
        """The JSON body of one POST /chat/completions request."""
        body: Dict[str, Any] = {
            "model": model,
            "messages": convert_messages(system, messages),
            self.max_tokens_field: int(max_tokens),
        }
        if tools:
            body["tools"] = to_openai_tools(tools)
            if tool_choice and tool_choice.get("type") == "none":
                body["tool_choice"] = "none"
        if effort:
            body["reasoning_effort"] = effort
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
            effort=effort,
        )
        data = json.dumps(body).encode("utf-8")
        url = f"{self.base_url}/chat/completions"
        headers = self._headers()
        last_err: Optional[str] = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    payload = resp.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as err:
                retry_after = err.headers.get("retry-after") if err.headers else None
                err_body = self._redact(http_error_body(err))
                last_err = f"HTTP {err.code}"
                if err.code == 402:
                    raise BudgetExceededError(f"OpenAI-compatible HTTP 402 (payment required): {err_body}") from err
                if err.code in RETRYABLE_HTTP_CODES and attempt < self.max_retries - 1:
                    sleep_s = float(retry_after) if retry_after and retry_after.isdigit() else (2.0**attempt)
                    time.sleep(min(sleep_s, MAX_RETRY_SLEEP_S))
                    continue
                raise RuntimeError(f"OpenAI-compatible HTTP {err.code} from {url}: {err_body}") from err
            except (urllib.error.URLError, TimeoutError, OSError) as err:
                last_err = self._redact(str(err))
                if attempt < self.max_retries - 1:
                    time.sleep(min(2.0**attempt, MAX_NETWORK_RETRY_SLEEP_S))
                    continue
                raise RuntimeError(f"OpenAI-compatible network error for {url}: {last_err}") from err
            try:
                raw = json.loads(payload)
            except json.JSONDecodeError:
                raise RuntimeError(
                    f"OpenAI-compatible reply from {url} is not JSON: {self._redact(payload[:200])!r}"
                ) from None
            return self._account(normalise_reply(raw, n_messages=len(messages)))
        raise RuntimeError(f"OpenAI-compatible API failed after retries: {last_err}")

    def _account(self, res: Dict[str, Any]) -> Dict[str, Any]:
        """Price the call, add it to the totals and attach the cost to the response."""
        in_tok = res["usage"]["input_tokens"]
        out_tok = res["usage"]["output_tokens"]
        cost = self.estimate_cost_usd(in_tok, out_tok)
        self.record_usage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_write_tokens=0,
            cache_read_tokens=0,
            cost_usd=cost,
        )
        res[COST_KEY] = cost
        return res
