"""Scripted stand-ins for model providers and HTTP, shared by the test modules.

`FakeHTTP` replaces urllib in-process; `LocalChatServer` is a real HTTP server on the
loopback interface for the OpenAI-compatible client. Nothing here reaches the network.
"""

from __future__ import annotations

import http.server
import io
import json
import threading
import urllib.request
from typing import Any, Callable, Dict, List, Optional
from unittest import mock

from blackwell_inversion.clients.base import COST_KEY, ModelClient


def text_reply(obj: Any, stop_reason: str = "end_turn", **usage: int) -> Dict[str, Any]:
    """A response holding one text block (a JSON dump of `obj` unless it is a string)."""
    text = obj if isinstance(obj, str) else json.dumps(obj)
    return {"content": [{"type": "text", "text": text}], "stop_reason": stop_reason, "usage": dict(usage)}


def tool_reply(name: str, tool_input: Dict[str, Any], call_id: str = "t1", **usage: int) -> Dict[str, Any]:
    """A response holding one tool call."""
    block = {"type": "tool_use", "id": call_id, "name": name, "input": tool_input}
    return {"content": [block], "stop_reason": "tool_use", "usage": dict(usage)}


class ScriptedClient(ModelClient):
    """A ModelClient whose replies come from a policy function of the call history.

    Each call charges `cost_per_call` and honours the spend cap, like a real provider.
    """

    def __init__(
        self,
        policy: Callable[[int, Dict[str, Any]], Dict[str, Any]],
        cost_per_call: float = 0.0,
        max_cost_usd: float = 1e9,
    ) -> None:
        super().__init__(max_cost_usd)
        self.policy = policy
        self.cost_per_call = cost_per_call
        self.calls: List[Dict[str, Any]] = []

    def create_message(self, **kwargs: Any) -> Dict[str, Any]:
        self.check_budget()
        snapshot = json.loads(json.dumps(kwargs))
        self.calls.append(snapshot)
        resp = self.policy(len(self.calls) - 1, snapshot)
        self.record_usage(
            input_tokens=int(resp.get("usage", {}).get("input_tokens", 0)),
            output_tokens=int(resp.get("usage", {}).get("output_tokens", 0)),
            cache_write_tokens=0,
            cache_read_tokens=0,
            cost_usd=self.cost_per_call,
        )
        resp[COST_KEY] = self.cost_per_call
        return resp

    @property
    def tool_choices(self) -> List[Optional[Dict[str, Any]]]:
        return [c.get("tool_choice") for c in self.calls]


def greedy_tool_policy(_index: int, call: Dict[str, Any]) -> Dict[str, Any]:
    """Ask for 50 random oracle queries whenever tools are allowed; answer with a fixed rule otherwise."""
    if call.get("tool_choice") == {"type": "none"}:
        return text_reply({"support": [0, 1, 2], "signs": [1, 1, 1]})
    n_assistant = sum(1 for m in call["messages"] if m["role"] == "assistant")
    return tool_reply("query_oracle_random", {"count": 50, "seed": n_assistant}, call_id=f"q{n_assistant}")


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class FakeHTTP:
    """Patch urllib.request.urlopen: record each request and answer from a queue.

    A queued item is a JSON-serialisable body, or an exception instance to raise.
    """

    def __init__(self, replies: List[Any]) -> None:
        self.replies = list(replies)
        self.requests: List[Dict[str, Any]] = []

    def __call__(self, req: urllib.request.Request, timeout: Optional[float] = None) -> _FakeResponse:
        self.requests.append(
            {
                "url": req.full_url,
                "method": req.get_method(),
                "headers": {k.lower(): v for k, v in req.header_items()},
                "body": json.loads(req.data.decode("utf-8")) if req.data else None,
            }
        )
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return _FakeResponse(json.dumps(reply).encode("utf-8"))

    def patch(self) -> Any:
        return mock.patch("urllib.request.urlopen", self)


class LocalChatServer:
    """A real HTTP server on 127.0.0.1 (ephemeral port) that plays an OpenAI-compatible API.

    Each POST is recorded as {"path", "headers", "body"} and answered by `responder`,
    called with (request index, parsed body, headers). It returns a JSON-serialisable
    body (status 200), or a tuple (status, body, extra headers) where a str or bytes body
    is sent verbatim. Use as a context manager; the server thread stops on exit.
    """

    def __init__(self, responder: Callable[[int, Any, Dict[str, str]], Any]) -> None:
        self.responder = responder
        self.requests: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 (http.server naming)
                raw = self.rfile.read(int(self.headers.get("content-length") or 0))
                try:
                    body: Any = json.loads(raw.decode("utf-8"))
                except ValueError:
                    body = raw.decode("utf-8", errors="replace")
                headers = {k.lower(): v for k, v in self.headers.items()}
                with outer._lock:
                    index = len(outer.requests)
                    outer.requests.append({"path": self.path, "headers": headers, "body": body})
                reply = outer.responder(index, body, headers)
                status, payload, extra = reply if isinstance(reply, tuple) else (200, reply, {})
                if isinstance(payload, str):
                    payload = payload.encode("utf-8")
                elif not isinstance(payload, bytes):
                    payload = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                for key, value in extra.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: Any) -> None:
                return None

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05})

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def bodies(self) -> List[Any]:
        return [r["body"] for r in self.requests]

    def __enter__(self) -> LocalChatServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()


def chat_reply(
    content: Optional[Any] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    finish_reason: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None,
    **message_extra: Any,
) -> Dict[str, Any]:
    """An OpenAI-compatible chat completion body."""
    message: Dict[str, Any] = {"role": "assistant", "content": content, **message_extra}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if finish_reason is None:
        finish_reason = "tool_calls" if tool_calls else "stop"
    choice = {"index": 0, "message": message, "finish_reason": finish_reason}
    body: Dict[str, Any] = {"id": "chatcmpl-test", "object": "chat.completion", "choices": [choice]}
    if usage is not None:
        body["usage"] = usage
    return body


def function_call(name: str, arguments: Any, call_id: Optional[str] = "c1") -> Dict[str, Any]:
    """One entry of `tool_calls`; `arguments` is sent as given (normally a JSON string)."""
    call: Dict[str, Any] = {"type": "function", "function": {"name": name, "arguments": arguments}}
    if call_id is not None:
        call["id"] = call_id
    return call
