"""The tool-calling agent loop shared by both experiments.

The agent gets up to three tools: `query_oracle_random` and `query_oracle` (metered
black-box queries, executed in this process) and, when allowed, `run_python` (code run
in the isolated sandbox). The loop alternates model turns and tool results until the
model answers without calling a tool or the turn budget is spent; an optional final
check can then grant revision turns (the harness_check protocols of Experiment 2).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .clients.base import COST_KEY, Message, ToolSchema
from .sandbox import IsolatedPythonSandbox, MeteredOracle

FINAL_TURN_NOTE = "Final turn: no further tool calls are allowed. Output your final JSON answer now."
NO_TOOLS = {"type": "none"}
MAX_LOGGED_CODE_CHARS = 8000
MAX_LOGGED_STDOUT_CHARS = 2000
FINAL_EXCERPT_CHARS = 240
# Errors returned to the model for a malformed tool call; such a call uses no query budget.
INPUT_NOT_OBJECT = "the arguments must be a JSON object."
COUNT_SEED_NOT_INT = "`count` and `seed` must be integers."

# Fields of the `run_agent_loop` result that both experiments copy into every record.
TELEMETRY_FIELDS = (
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


def build_tool_schemas(allow_python: bool, n: int) -> List[ToolSchema]:
    """Tool definitions offered to the model for inputs in {-1, +1}^n."""
    tools: List[ToolSchema] = [
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


def extract_text_blocks(response: Dict[str, Any]) -> str:
    """Concatenated text blocks of a model response."""
    parts: List[str] = []
    for blk in response.get("content", []):
        if isinstance(blk, dict) and blk.get("type") == "text":
            parts.append(str(blk.get("text", "")))
    return "\n".join(parts).strip()


def _int_or_none(value: Any) -> Optional[int]:
    """int(value) as the loop has always applied it to tool arguments, or None where that fails."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _tool_uses(content: List[Any]) -> List[Dict[str, Any]]:
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]


def _append_user_note(messages: List[Message], text: str) -> None:
    """Add a text block to the pending user turn, opening one if the last turn is the model's.

    A new user turn first answers every unanswered tool call of the preceding model turn
    (possible when that turn was cut at the token cap); the API requires those results.
    """
    note_blk = {"type": "text", "text": text}
    last = messages[-1] if messages else None
    if last is not None and last.get("role") == "user":
        if isinstance(last["content"], list):
            last["content"].append(note_blk)
        else:
            last["content"] = [{"type": "text", "text": str(last["content"])}, note_blk]
        return
    last_content = last.get("content", []) if last is not None else []
    pending = [
        {
            "type": "tool_result",
            "tool_use_id": b.get("id", "unknown"),
            "content": json.dumps({"ok": False, "error": "Not executed: no tool calls left."}),
        }
        for b in _tool_uses(last_content if isinstance(last_content, list) else [])
    ]
    messages.append({"role": "user", "content": pending + [note_blk]})


@dataclass
class _Telemetry:
    """Per-trial token, cost and stop-reason accounting across all model calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    truncated_turns: int = 0
    stop_reasons: List[str] = field(default_factory=list)

    def add(self, resp: Dict[str, Any]) -> str:
        """Account for one response and return its stop reason."""
        usage = resp.get("usage", {})
        self.input_tokens += int(usage.get("input_tokens", 0))
        self.output_tokens += int(usage.get("output_tokens", 0))
        self.cache_write_tokens += int(usage.get("cache_creation_input_tokens", 0))
        self.cache_read_tokens += int(usage.get("cache_read_input_tokens", 0))
        out_det = usage.get("output_tokens_details") or {}
        if isinstance(out_det, dict):
            self.thinking_tokens += int(out_det.get("thinking_tokens", 0))
        self.cost_usd += float(resp.get(COST_KEY, 0.0))
        stop_reason = str(resp.get("stop_reason", ""))
        self.stop_reasons.append(stop_reason)
        if stop_reason == "max_tokens":
            self.truncated_turns += 1
        return stop_reason


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
    log_tool_inputs: bool = False,
    final_check: Optional[Callable[[str], Optional[str]]] = None,
    revision_turns: int = 1,
) -> Dict[str, Any]:
    """Run one agent episode and return its final text and telemetry.

    The tool list is identical on every turn (prompt caching and thinking-block binding
    depend on it); the last turn forbids tool calls through `tool_choice` instead.

    log_tool_inputs: also return the Python code and query sizes of every tool call.
    final_check: called with the agent's final text after the loop; if it returns a
        note, the note is sent as a new user turn and the reply to the revision becomes
        the final answer.
    revision_turns: turns granted after a failed final_check. With 1, that single turn
        is tool-free; with R > 1 the first R - 1 turns may call tools and the last may not.
    """
    tools = build_tool_schemas(allow_python=allow_python, n=oracle.n)
    tool_log: List[Dict[str, Any]] = []
    tool_sequence: List[str] = []
    messages: List[Message] = [{"role": "user", "content": user_prompt}]
    telemetry = _Telemetry()
    turns_used = 0
    final_text = ""
    t0 = time.monotonic()

    def call_model(tool_choice: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return client.create_message(
            model=model,
            system=system,
            messages=messages,
            max_tokens=max_tokens_per_turn,
            tools=tools,
            tool_choice=dict(tool_choice) if tool_choice is not None else None,
            thinking_budget=thinking_budget,
            thinking_mode=thinking_mode,
            effort=effort,
        )

    def execute_tools(tool_uses: List[Dict[str, Any]], turn_idx: int) -> List[Dict[str, Any]]:
        tool_results: List[Dict[str, Any]] = []
        for tu in tool_uses:
            tname = tu.get("name", "")
            tinput = tu.get("input", {}) or {}
            if not isinstance(tinput, dict):
                out_obj = {"ok": False, "error": f"Invalid input for tool {tname!r}: {INPUT_NOT_OBJECT}"}
            elif tname == "query_oracle_random":
                count, seed = _int_or_none(tinput.get("count", 1)), _int_or_none(tinput.get("seed", 0))
                if count is None or seed is None:
                    out_obj = {"ok": False, "error": f"Invalid input for tool {tname!r}: {COUNT_SEED_NOT_INT}"}
                else:
                    out_obj = oracle.query_random(count=count, seed=seed)
            elif tname == "query_oracle":
                out_obj = oracle.query(tinput.get("inputs", []))
            elif tname == "run_python" and allow_python and sandbox is not None:
                out_obj = sandbox.run_python(str(tinput.get("code", "")))
            else:
                out_obj = {"ok": False, "error": f"Tool {tname!r} is unavailable."}
            tool_sequence.append(f"{tname}:{'ok' if out_obj.get('ok') else 'err'}")
            if log_tool_inputs:
                entry: Dict[str, Any] = {"turn": turn_idx, "tool": tname, "ok": bool(out_obj.get("ok"))}
                if tname == "run_python":
                    code = tinput.get("code", "") if isinstance(tinput, dict) else tinput
                    entry["code"] = str(code)[:MAX_LOGGED_CODE_CHARS]
                    entry["stdout"] = str(out_obj.get("stdout", ""))[:MAX_LOGGED_STDOUT_CHARS]
                else:
                    entry["n_outputs"] = len(out_obj.get("outputs", []) or [])
                    entry["queries_used_after"] = oracle.queries_used
                tool_log.append(entry)
            tool_results.append(
                {"type": "tool_result", "tool_use_id": tu.get("id", "unknown"), "content": json.dumps(out_obj)}
            )
        return tool_results

    for turn_idx in range(max_turns):
        turns_used = turn_idx + 1
        is_last_turn = turn_idx == max_turns - 1
        if is_last_turn and turn_idx > 0:
            _append_user_note(messages, FINAL_TURN_NOTE)
        resp = call_model(NO_TOOLS if is_last_turn else None)
        stop_reason = telemetry.add(resp)
        content = resp.get("content", [])
        messages.append({"role": "assistant", "content": content})
        tool_uses = _tool_uses(content)
        if not tool_uses or stop_reason != "tool_use":
            final_text = extract_text_blocks(resp)
            break
        messages.append({"role": "user", "content": execute_tools(tool_uses, turn_idx)})
        if is_last_turn:
            final_text = extract_text_blocks(resp)

    harness_check = ""
    if final_check is not None:
        note = final_check(final_text)
        harness_check = "pass" if note is None else "revise"
        if note is not None:
            _append_user_note(messages, note)
            tool_sequence.append("harness_check:revise")
            n_rev = max(1, int(revision_turns))
            for rev_idx in range(n_rev):
                last_rev = rev_idx == n_rev - 1
                if last_rev and rev_idx > 0:
                    _append_user_note(messages, FINAL_TURN_NOTE)
                resp = call_model(NO_TOOLS if last_rev else None)
                stop_reason = telemetry.add(resp)
                turns_used += 1
                content = resp.get("content", [])
                tool_uses = _tool_uses(content)
                if last_rev or not tool_uses or stop_reason != "tool_use":
                    final_text = extract_text_blocks(resp)
                    break
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": execute_tools(tool_uses, turns_used - 1)})

    return {
        "harness_check": harness_check,
        "tool_log": tool_log,
        "final_text": final_text,
        "final_excerpt": final_text[:FINAL_EXCERPT_CHARS],
        "tool_sequence": tool_sequence,
        "stop_reasons": telemetry.stop_reasons,
        "truncated_turns": telemetry.truncated_turns,
        "turns": turns_used,
        "input_tokens": telemetry.input_tokens,
        "output_tokens": telemetry.output_tokens,
        "thinking_tokens": telemetry.thinking_tokens,
        "cache_read_tokens": telemetry.cache_read_tokens,
        "cache_write_tokens": telemetry.cache_write_tokens,
        "cost_usd": round(telemetry.cost_usd, 6),
        "queries_used": oracle.queries_used,
        "python_calls": sandbox.exec_count if sandbox is not None else 0,
        "wall_s": round(time.monotonic() - t0, 4),
    }
