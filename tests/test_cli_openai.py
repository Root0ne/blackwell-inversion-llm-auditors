"""Whole trials through the command line with `--provider openai` against a scripted local server.

The "model" is a function of the request body served on 127.0.0.1: it sees exactly what
a real server would see, and its replies go through the client, the agent loop, the
sandbox, the oracle, the parser and the scorer into the JSONL log.
"""

from __future__ import annotations

import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from blackwell_inversion import cli
from blackwell_inversion.generators import generate_blackwell_inv_instance

from .fakes import LocalChatServer, chat_reply, function_call
from .test_agent_protocols import EXP1_FIELDS, EXP2_FIELDS
from .test_openai_compat import hermetic_env

USAGE = {"prompt_tokens": 1000, "completion_tokens": 100}


def _assistant_turns(body: Dict[str, Any]) -> int:
    return sum(1 for m in body["messages"] if m["role"] == "assistant")


def correlation_model(_i: int, body: Dict[str, Any], _headers: Dict[str, str]) -> Dict[str, Any]:
    """Queries 80 fixed random inputs, then answers with the three most correlated coordinates."""
    if _assistant_turns(body) == 0:
        rng = random.Random(7)
        inputs = [[rng.choice((-1, 1)) for _ in range(40)] for _ in range(80)]
        return chat_reply(None, [function_call("query_oracle", json.dumps({"inputs": inputs}), None)], usage=USAGE)
    call = next(m for m in body["messages"] if m["role"] == "assistant")["tool_calls"][0]
    inputs = json.loads(call["function"]["arguments"])["inputs"]
    result = next(m for m in body["messages"] if m["role"] == "tool")
    assert result["tool_call_id"] == call["id"] == "call_1_0"  # the synthesised id was replayed
    outputs = json.loads(result["content"])["outputs"]
    corr = [sum(x[i] * (2 * y - 1) for x, y in zip(inputs, outputs)) for i in range(40)]
    top = sorted(sorted(range(40), key=lambda i: -abs(corr[i]))[:3])
    answer = {"support": top, "signs": [1 if corr[i] > 0 else -1 for i in top]}
    return chat_reply(f"Final answer: {json.dumps(answer)}", usage=USAGE, reasoning_content="(hidden reasoning)")


class TestOpenAIProviderThroughTheCLI(unittest.TestCase):
    def setUp(self) -> None:
        hermetic_env(self)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name) / "log.jsonl"

    def run_cli(self, responder: Any, argv: List[str]) -> tuple:
        server = LocalChatServer(responder)
        log = io.StringIO()
        with server, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            rc = cli.main(argv + ["--provider", "openai", "--base-url", server.base_url, "--models", "local-model",
                                  "--input-price-per-mtok", "1", "--output-price-per-mtok", "2", "--out", str(self.out)])
        self.assertEqual(rc, 0, log.getvalue())
        records = [json.loads(line) for line in self.out.read_text(encoding="utf-8").splitlines()]
        return records, server, log.getvalue()

    def test_exp2_trial_is_solved_and_scored(self) -> None:
        records, server, _ = self.run_cli(correlation_model, [
            "--experiment", "exp2", "--variants", "wb_clean", "--budget-tracks", "matched_medium_budget", "--exp2-seeds", "1"])
        inst = generate_blackwell_inv_instance(n=40, support_size=3, variant="wb_clean", seed=5000)
        self.assertEqual([r["mode"] for r in records], ["BB", "WB"])
        for rec in records:
            self.assertEqual(list(rec), EXP2_FIELDS + ["run_key"])
            self.assertEqual((rec["pred_support"], rec["pred_signs"]), (list(inst.true_support), list(inst.true_signs)))
            self.assertEqual((rec["parse_ok"], rec["exact_rule"], rec["heldout_acc"]), (True, 1.0, 1.0))
            self.assertEqual((rec["turns"], rec["queries_used"], rec["python_calls"]), (2, 80, 0))
            self.assertEqual(rec["stop_reasons"], ["tool_use", "end_turn"])
            self.assertEqual(rec["tool_sequence"], ["query_oracle:ok"])
            self.assertEqual((rec["input_tokens"], rec["output_tokens"]), (2000, 200))
            self.assertAlmostEqual(rec["cost_usd"], 2 * (1000 * 1 + 100 * 2) / 1e6)
            self.assertNotIn("hidden reasoning", rec["final_excerpt"])
        self.assertEqual(records[0]["run_key"], "exp2|local-model|tm=adaptive|ef=medium|matched_medium_budget|BB|blackbox|s5000")
        first = server.requests[0]["body"]
        self.assertEqual(first["model"], "local-model")
        self.assertEqual(first["reasoning_effort"], "medium")
        self.assertEqual(first["max_tokens"], 4096)
        self.assertEqual([t["function"]["name"] for t in first["tools"]], ["query_oracle_random", "query_oracle", "run_python"])

    def test_exp1_trials_with_tool_calls_under_finish_reason_stop(self) -> None:
        def model(_i: int, body: Dict[str, Any], _h: Dict[str, str]) -> Dict[str, Any]:
            if _assistant_turns(body) == 0:
                return chat_reply(None, [function_call("query_oracle_random", '{"count": 4, "seed": 3}', "q")],
                                  finish_reason="stop")
            return chat_reply('{"decision": "PLANTED", "confidence": 0.6, "rationale": "scripted"}')

        records, _, _ = self.run_cli(model, ["--experiment", "exp1", "--regimes", "vbb_hard", "--substrates", "code_clean",
                                             "--exp1-pairs", "1", "--exp1-skip-bb"])
        self.assertEqual(len(records), 2)
        for rec in records:
            self.assertEqual(list(rec), EXP1_FIELDS + ["run_key"])
            self.assertEqual((rec["decision_label"], rec["parse_ok"]), ("PLANTED", True))
            self.assertEqual(rec["correct"], rec["planted"])
            self.assertEqual((rec["queries_used"], rec["turns"]), (4, 2))
            self.assertEqual(rec["tool_sequence"], ["query_oracle_random:ok"])
        self.assertEqual(sorted(r["planted"] for r in records), [0, 1])

    def test_malformed_tool_arguments_get_an_error_and_cost_no_queries(self) -> None:
        bad_calls = ["{count: 5", '{"count": "five", "seed": 1}', '{"count": 5, "seed": "x"}', "[5]"]

        def model(_i: int, body: Dict[str, Any], _h: Dict[str, str]) -> Dict[str, Any]:
            turn = _assistant_turns(body)
            if body.get("tool_choice") == "none":
                return chat_reply('{"support": [0, 1, 2], "signs": [1, 1, 1]}')
            return chat_reply("trying", [function_call("query_oracle_random", bad_calls[turn], f"b{turn}")])

        records, server, _ = self.run_cli(model, [
            "--experiment", "exp2", "--only-bb", "--budget-tracks", "matched_medium_budget", "--exp2-seeds", "1"])
        rec = records[0]
        self.assertEqual(rec["queries_used"], 0)
        self.assertEqual(rec["tool_sequence"], ["query_oracle_random:err"] * 4)
        self.assertEqual((rec["turns"], rec["parse_ok"]), (5, True))
        last = server.requests[-1]["body"]["messages"]
        replayed = [m["tool_calls"][0]["function"]["arguments"] for m in last if m["role"] == "assistant"]
        self.assertEqual(replayed, bad_calls)  # raw arguments are sent back unchanged
        errors = [json.loads(m["content"]) for m in last if m["role"] == "tool"]
        self.assertEqual([e["ok"] for e in errors], [False] * 4)
        self.assertIn("must be a JSON object", errors[0]["error"])
        self.assertIn("must be integers", errors[1]["error"])
        self.assertIn("must be integers", errors[2]["error"])
        self.assertIn("must be a JSON object", errors[3]["error"])

    def test_model_that_never_calls_a_tool(self) -> None:
        records, server, _ = self.run_cli(lambda *_: chat_reply("I believe the rule uses x3, x7 and x9."), [
            "--experiment", "exp2", "--only-bb", "--budget-tracks", "matched_total_budget", "--exp2-seeds", "1"])
        rec = records[0]
        self.assertEqual(len(server.requests), 1)
        self.assertEqual((rec["turns"], rec["queries_used"], rec["tool_sequence"]), (1, 0, []))
        self.assertEqual((rec["parse_ok"], rec["heldout_acc"], rec["pred_support"]), (False, 0.5, []))
        self.assertEqual((rec["input_tokens"], rec["cost_usd"]), (0, 0.0))  # the server sent no usage

    def test_an_uninterpretable_reply_writes_no_record(self) -> None:
        server = LocalChatServer(lambda *_: {"choices": []})
        log = io.StringIO()
        with server, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), self.assertRaises(RuntimeError):
            cli.main(["--experiment", "exp2", "--only-bb", "--exp2-seeds", "1", "--budget-tracks", "matched_total_budget",
                      "--provider", "openai", "--base-url", server.base_url, "--models", "m", "--out", str(self.out)])
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
