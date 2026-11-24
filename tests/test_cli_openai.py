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
import socket
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

from blackwell_inversion import cli
from blackwell_inversion.generators import generate_blackwell_inv_instance

from .fakes import LocalChatServer, chat_reply, function_call
from .test_agent_protocols import EXP1_FIELDS, EXP2_FIELDS
from .test_openai_compat import SLEEP, hermetic_env

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


class _CLITest(unittest.TestCase):
    def setUp(self) -> None:
        hermetic_env(self)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name) / "log.jsonl"

    def run_cli(self, responder: Any, argv: List[str], status: int = 0) -> tuple:
        server = LocalChatServer(responder)
        log = io.StringIO()
        with server, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), mock.patch(SLEEP):
            rc = cli.main(argv + ["--provider", "openai", "--base-url", server.base_url, "--models", "local-model",
                                  "--input-price-per-mtok", "1", "--output-price-per-mtok", "2", "--out", str(self.out)])
        self.assertEqual(rc, status, log.getvalue())
        lines = self.out.read_text(encoding="utf-8").splitlines() if self.out.exists() else []
        return [json.loads(line) for line in lines], server, log.getvalue()


class TestOpenAIProviderThroughTheCLI(_CLITest):
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

    def test_a_failing_trial_is_reported_and_the_run_continues(self) -> None:
        # Each server breaks the first trial in its own way; the second trial succeeds.
        context_error = (400, json.dumps({"error": {"message": "This model's maximum context length is 8192 tokens"}}), {})
        no_message = chat_reply("x")
        no_message["choices"][0]["message"] = None
        failures = {
            "uninterpretable": ({"choices": []}, 1, "OpenAI-compatible reply has no choices."),
            "not JSON": ((200, "<html>\nproxy login\n</html>", {}), 1, "is not JSON"),
            "message null": (no_message, 1, "OpenAI-compatible reply has no message."),
            "context length": (context_error, 1, "HTTP 400"),
            "persistent 500": ((500, "boom", {}), 5, "HTTP 500"),  # retried, then given up
        }
        for name, (bad, n_bad, message) in failures.items():
            with self.subTest(name):
                self.out.unlink(missing_ok=True)

                def model(i: int, _body: Any, _h: Dict[str, str], bad: Any = bad, n_bad: int = n_bad) -> Any:
                    return bad if i < n_bad else chat_reply('{"support": [0, 1, 2], "signs": [1, 1, 1]}')

                records, server, log = self.run_cli(model, [
                    "--experiment", "exp2", "--only-bb", "--exp2-seeds", "2", "--budget-tracks", "matched_total_budget"],
                    status=cli.EXIT_TRIAL_ERRORS)
                errors = [line for line in log.splitlines() if line.startswith("[TRIAL ERROR]")]
                self.assertEqual(len(errors), 1, log)
                self.assertTrue(errors[0].startswith(
                    "[TRIAL ERROR] exp2|local-model|tm=adaptive|ef=medium|matched_total_budget|BB|blackbox|s5000: "), log)
                self.assertIn(message, errors[0])
                self.assertNotIn("Traceback", log)
                self.assertEqual([r["seed"] for r in records], [5001])
                self.assertIn("executed=1, skipped_resumed=0, errors=1", log)

    def test_a_dead_server_stops_the_run_with_one_line_per_trial(self) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            url = f"http://127.0.0.1:{sock.getsockname()[1]}/v1"
        log = io.StringIO()
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), mock.patch(SLEEP):
            rc = cli.main(["--experiment", "exp2", "--only-bb", "--exp2-seeds", "8", "--budget-tracks", "matched_total_budget",
                           "--provider", "openai", "--base-url", url, "--models", "m", "--out", str(self.out)])
        self.assertEqual(rc, cli.EXIT_TRIAL_ERRORS)
        errors = [line for line in log.getvalue().splitlines() if line.startswith("[TRIAL ERROR]")]
        self.assertEqual(len(errors), cli.DEFAULT_MAX_CONSECUTIVE_ERRORS)
        for line in errors:
            self.assertIn(f"cannot reach the OpenAI-compatible server at {url} after 5 attempts", line)
        self.assertIn(f"[STOPPED — {cli.DEFAULT_MAX_CONSECUTIVE_ERRORS} CONSECUTIVE TRIAL ERRORS]", log.getvalue())
        self.assertNotIn("Traceback", log.getvalue())
        self.assertFalse(self.out.exists())

    def test_setup_errors_stop_before_any_trial(self) -> None:
        base = ["--preset", "smoke", "--provider", "openai", "--base-url", "http://127.0.0.1:9/v1", "--out", str(self.out)]
        for argv, message in (
            (base + ["--models", "m", "--api-key-env", "UNSET_TEST_KEY_VAR"], "--api-key-env names UNSET_TEST_KEY_VAR"),
            (base + ["--models", "m", "--api-key-env", "UNSET_TEST_KEY_VAR", "--workers", "4"], "UNSET_TEST_KEY_VAR"),
            (base, "--provider openai needs --models"),
            (base + ["--models", "m", "--max-consecutive-errors", "-1"], "--max-consecutive-errors"),
        ):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as ctx:
                cli.main(argv)
            self.assertEqual(ctx.exception.code, 2)
            self.assertIn(message, err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())
            self.assertFalse(self.out.exists())


def strict_server(check: Any, extra_call: Dict[str, Any], extra_message: Dict[str, Any]) -> Any:
    """A server that rejects a request whose earlier assistant turns lack what it attached to them."""

    def model(_i: int, body: Dict[str, Any], _h: Dict[str, str]) -> Any:
        for msg in body["messages"]:
            if msg["role"] == "assistant" and not check(msg):
                return (400, json.dumps({"error": {"message": "Function call is missing a thought_signature"}}), {})
        if body.get("tool_choice") == "none":
            return chat_reply('{"support": [0, 1, 2], "signs": [1, 1, 1]}')
        turn = _assistant_turns(body)
        call = function_call("query_oracle_random", '{"count": 5}', f"c{turn}") | extra_call
        return chat_reply(None, [call], **extra_message)

    return model


class TestReplayToStrictServers(_CLITest):
    """Whole trials against servers that need their own fields back (extra_content, reasoning)."""

    ARGV = ["--experiment", "exp2", "--only-bb", "--exp2-seeds", "1", "--budget-tracks", "matched_medium_budget"]

    def _check_trial(self, records: List[Dict[str, Any]], server: LocalChatServer) -> None:
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["tool_sequence"], ["query_oracle_random:ok"] * 4)
        self.assertEqual((records[0]["turns"], records[0]["queries_used"], records[0]["parse_ok"]), (5, 20, True))
        self.assertEqual(len(server.requests), 5)

    def test_thought_signatures_on_tool_calls(self) -> None:
        sig = {"extra_content": {"google": {"thought_signature": "c2ln"}}}
        records, server, _ = self.run_cli(
            strict_server(lambda m: all(c.get("extra_content") == sig["extra_content"] for c in m.get("tool_calls", [])), sig, {}),
            self.ARGV)
        self._check_trial(records, server)
        last = server.requests[-1]["body"]["messages"]
        self.assertEqual([c["id"] for m in last if m["role"] == "assistant" for c in m["tool_calls"]], ["c0", "c1", "c2", "c3"])

    def test_reasoning_content_is_sent_back(self) -> None:
        fields = {"reasoning_content": "I should sample.", "reasoning_details": [{"type": "reasoning.text", "text": "r"}]}
        records, server, _ = self.run_cli(
            strict_server(lambda m: all(m.get(k) == v for k, v in fields.items()), {}, fields), self.ARGV)
        self._check_trial(records, server)
        self.assertNotIn("I should sample", records[0]["final_excerpt"])


if __name__ == "__main__":
    unittest.main()
