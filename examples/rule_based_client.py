"""A toy custom client: a rule-based "model" that plugs into the benchmark via --client.

    python -m blackwell_inversion --client examples.rule_based_client:make_client \
        --preset smoke --models toy-correlation --out toy.jsonl
    python -m blackwell_inversion.analysis.summarize toy.jsonl

`--client MODULE:FACTORY` imports MODULE, calls FACTORY(args) with the parsed command
line and uses the returned `ModelClient`. Replace the body of `create_message` with a
call to your own model (an in-process model, a proprietary API, ...); everything else
(prompts, tools, sandbox, scoring, logging) stays the same. The contract is documented
in `blackwell_inversion/clients/base.py`.

This client needs no network and only the standard library. In Exp 2 it queries the
oracle on fixed random inputs, picks the three coordinates most correlated with the
output and answers with them; it ignores the white-box artifact. In Exp 1 it always
answers CLEAN. It is a pipeline example, not a baseline of the paper.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional

from blackwell_inversion.clients import COST_KEY, ModelClient

EXP2_MARKER = 'Conclude with JSON: {"support": ['
INPUT_SEED = 0


def _prompt_int(pattern: str, text: str, default: int) -> int:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else default


def _reply(blocks: List[Dict[str, Any]], stop_reason: str) -> Dict[str, Any]:
    return {"content": blocks, "stop_reason": stop_reason, "usage": {"input_tokens": 0, "output_tokens": 0}, COST_KEY: 0.0}


class RuleBasedClient(ModelClient):
    """Answers from oracle correlations; costs nothing."""

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
        self.check_budget()
        prompt = str(messages[0]["content"])
        if EXP2_MARKER not in prompt:
            answer = {"decision": "CLEAN", "confidence": 0.5, "rationale": "toy client"}
            return self._account(_reply([{"type": "text", "text": json.dumps(answer)}], "end_turn"))

        n = _prompt_int(r"\{-1, \+1\}\^(\d+)", prompt, 40)
        budget = _prompt_int(r"total budget (\d+) queries", prompt, 80)
        asked = [m for m in messages if m["role"] == "assistant"]
        tools_allowed = tool_choice != {"type": "none"}
        if not asked and tools_allowed:
            rng = random.Random(INPUT_SEED)
            inputs = [[rng.choice((-1, 1)) for _ in range(n)] for _ in range(budget)]
            call = {"type": "tool_use", "id": "toy_q1", "name": "query_oracle", "input": {"inputs": inputs}}
            return self._account(_reply([call], "tool_use"))
        return self._account(_reply([{"type": "text", "text": json.dumps(self._fit(messages, n))}], "end_turn"))

    @staticmethod
    def _fit(messages: List[Dict[str, Any]], n: int) -> Dict[str, Any]:
        """Top-3 correlated coordinates from our query and the oracle's answer, or a fixed guess."""
        guess = {"support": [0, 1, 2], "signs": [1, 1, 1]}
        if len(messages) < 3 or messages[1]["role"] != "assistant":
            return guess
        inputs = next((b["input"]["inputs"] for b in messages[1]["content"] if b.get("type") == "tool_use"), None)
        results = [b for b in messages[2]["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]
        if inputs is None or not results:
            return guess
        outputs = json.loads(results[0]["content"]).get("outputs") or []
        if not outputs:
            return guess
        corr = [sum(x[i] * (2 * y - 1) for x, y in zip(inputs, outputs)) for i in range(n)]
        top = sorted(sorted(range(n), key=lambda i: -abs(corr[i]))[:3])
        return {"support": top, "signs": [1 if corr[i] >= 0 else -1 for i in top]}

    def _account(self, reply: Dict[str, Any]) -> Dict[str, Any]:
        self.record_usage(input_tokens=0, output_tokens=0, cache_write_tokens=0, cache_read_tokens=0, cost_usd=0.0)
        return reply


def make_client(args: Any) -> RuleBasedClient:
    """The factory `--client` calls with the parsed command line."""
    return RuleBasedClient(max_cost_usd=args.max_cost_usd)
