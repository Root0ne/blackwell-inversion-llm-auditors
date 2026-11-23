"""The contract between the agent loop and a model provider.

A client turns one conversation state into one model reply. The agent loop
(`blackwell_inversion.agent.run_agent_loop`) calls only `create_message`; the command
line driver additionally reads `total_cost_usd` to report spend. Every provider client
subclasses `ModelClient`, which also keeps the spend counters and enforces the
`--max-cost-usd` cap.

Request (keyword arguments of `create_message`), in Anthropic Messages API form:

    model           provider model id
    system          system prompt
    messages        alternating user/assistant turns; content is a string or a list of
                    blocks {"type": "text", ...}, {"type": "tool_use", "id", "name",
                    "input"} and {"type": "tool_result", "tool_use_id", "content"}
    max_tokens      output token cap for this turn
    tools           tool schemas {"name", "description", "input_schema"} (JSON Schema)
    tool_choice     None, or {"type": "none"} on a turn where tools must not be called
    thinking_budget, thinking_mode, effort
                    reasoning settings; each client maps them to its provider

Response (the dict `create_message` returns):

    content         list of blocks as above; the loop reads "text" and "tool_use" blocks
                    and stores the list verbatim as the assistant turn, so a client may
                    add private blocks it needs to replay on the next call
    stop_reason     "tool_use" when the reply calls tools, "max_tokens" when it was cut
                    at the token cap, anything else (e.g. "end_turn") otherwise
    usage           {"input_tokens", "output_tokens", optional "cache_read_input_tokens",
                    "cache_creation_input_tokens", and "output_tokens_details":
                    {"thinking_tokens"}}
    "_cost_usd"     (`COST_KEY`) estimated USD cost of this call, see `pricing.estimate_cost_usd`

A client must raise `BudgetExceededError` before sending a request once its cumulative
spend has reached `max_cost_usd`; the driver stops scheduling trials when it sees it.
"""

from __future__ import annotations

import abc
import os
import threading
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional

COST_KEY = "_cost_usd"

Message = Dict[str, Any]
ToolSchema = Dict[str, Any]


class BudgetExceededError(RuntimeError):
    """Raised before an API request once the --max-cost-usd spend cap has been reached."""


def resolve_api_key(explicit: Optional[str], env_var: str) -> str:
    """Return the explicit key, else the environment variable, else the entry in ~/.env."""
    return explicit or os.environ.get(env_var, "").strip() or _key_from_dotenv(env_var)


def http_error_body(err: urllib.error.HTTPError, limit: int = 500) -> str:
    """The first `limit` characters of an HTTP error response, for the exception message."""
    try:
        return err.read().decode("utf-8", errors="replace")[:limit]
    except Exception:
        return ""


def _key_from_dotenv(var_name: str) -> str:
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


class ModelClient(abc.ABC):
    """Base class of all provider clients: the `create_message` contract plus spend accounting.

    Counters are updated under a lock so one client can serve several worker threads.
    """

    def __init__(self, max_cost_usd: float) -> None:
        self.max_cost_usd = float(max_cost_usd)
        self._lock = threading.Lock()
        self.total_cost_usd = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_write_tokens = 0

    @abc.abstractmethod
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
        """Send one request and return the normalised response described in the module docstring."""

    def check_budget(self) -> None:
        """Raise BudgetExceededError if the cumulative spend has reached the cap."""
        with self._lock:
            if self.total_cost_usd >= self.max_cost_usd:
                raise BudgetExceededError(
                    f"Spend limit reached: ${self.total_cost_usd:.4f} >= ${self.max_cost_usd:.4f}"
                )

    def record_usage(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_write_tokens: int,
        cache_read_tokens: int,
        cost_usd: float,
    ) -> None:
        """Add one call's billed tokens and cost to the running totals."""
        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cache_write_tokens += cache_write_tokens
            self.total_cache_read_tokens += cache_read_tokens
            self.total_cost_usd += cost_usd
