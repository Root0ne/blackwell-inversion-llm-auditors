"""The agent loop, the Exp 2 protocols and their turn and query budgets, and the record schema."""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict

from blackwell_inversion.agent import FINAL_TURN_NOTE, build_tool_schemas, run_agent_loop
from blackwell_inversion.clients import BudgetExceededError
from blackwell_inversion.exp1 import run_exp1_trial
from blackwell_inversion.exp2 import (
    BUDGET_TRACK_TURNS,
    EXP2_CHECK_K,
    EXP2_REVISION_TURNS,
    VERIFY_FIRST_NOTE,
    run_exp2_trial,
    turns_for,
)
from blackwell_inversion.generators import generate_blackwell_inv_instance
from blackwell_inversion.sandbox import IsolatedPythonSandbox, MeteredOracle

from .fakes import ScriptedClient, greedy_tool_policy, text_reply, tool_reply

NO_TOOLS = {"type": "none"}

EXP2_FIELDS = [
    "experiment", "model", "thinking_budget", "thinking_mode", "effort", "task_type", "variant", "mode",
    "budget_track", "seed", "instance_id", "n", "q_max", "true_support", "true_signs", "decoy_support",
    "pred_support", "pred_signs", "parse_ok", "heldout_acc", "support_f1", "exact_support", "exact_rule",
    "matches_decoy", "turns", "input_tokens", "output_tokens", "thinking_tokens", "cache_read_tokens",
    "cache_write_tokens", "truncated_turns", "stop_reasons", "cost_usd", "queries_used", "python_calls",
    "wall_s", "tool_sequence", "final_excerpt",
]
EXP1_FIELDS = [
    "experiment", "model", "thinking_budget", "thinking_mode", "effort", "family", "regime", "substrate", "mode",
    "track", "seed", "instance_id", "n", "m", "k", "eta", "planted", "decision_int", "decision_label", "parse_ok",
    "correct", "turns", "input_tokens", "output_tokens", "thinking_tokens", "cache_read_tokens",
    "cache_write_tokens", "truncated_turns", "stop_reasons", "cost_usd", "queries_used", "python_calls",
    "wall_s", "tool_sequence", "final_excerpt",
]
CHECK_FIELDS = ["check_queries", "check_result", "first_pred_support", "first_pred_signs", "first_parse_ok"]


class _ScriptedClient:
    """Answers with the decoy rule first; after a failed check it uses tools when allowed, then answers correctly.

    Duck-typed on purpose: the agent loop needs nothing but `create_message`.
    """

    def __init__(self, wrong: dict, right: dict) -> None:
        self.wrong, self.right = wrong, right
        self.tool_choices: list = []

    def create_message(self, *, messages, tool_choice=None, **_kw):
        self.tool_choices.append(tool_choice)
        n_assistant = sum(1 for m in messages if m.get("role") == "assistant")

        def text(obj: dict) -> dict:
            return {"content": [{"type": "text", "text": json.dumps(obj)}], "stop_reason": "end_turn", "usage": {}}

        def tool(name: str, inp: dict) -> dict:
            block = {"type": "tool_use", "id": f"t{n_assistant}", "name": name, "input": inp}
            return {"content": [block], "stop_reason": "tool_use", "usage": {}}

        if n_assistant == 0:
            return text(self.wrong)
        if tool_choice == {"type": "none"}:
            return text(self.right)
        if n_assistant == 1:
            return tool("run_python", {"code": "import json\nprint(len(json.load(open('queries.json'))['records']))"})
        return tool("query_oracle_random", {"count": 5, "seed": 1})


class TestAnswerCheckProtocols(unittest.TestCase):
    def _run(self, protocol: str):
        inst = generate_blackwell_inv_instance(n=40, support_size=3, variant="wb_inv_code", seed=5000)
        wrong = {"support": list(inst.decoy_support), "signs": [1, 1, 1]}
        right = {"support": list(inst.true_support), "signs": list(inst.true_signs)}
        client = _ScriptedClient(wrong, right)
        rec = run_exp2_trial(
            client=client,
            model="scripted",
            variant="wb_inv_code",
            mode="WB",
            budget_track="matched_medium_budget",
            seed=5000,
            protocol=protocol,
            log_tool_inputs=True,
        )
        return inst, client, rec

    def test_tool_free_revision_turn(self) -> None:
        inst, client, rec = self._run("harness_check")
        self.assertEqual(rec["check_result"], "mismatch")
        self.assertEqual(rec["first_pred_support"], list(inst.decoy_support))
        self.assertEqual(rec["turns"], 2)
        self.assertEqual(client.tool_choices, [None, {"type": "none"}])
        self.assertEqual(rec["pred_support"], list(inst.true_support))
        self.assertEqual(rec["queries_used"], 0)

    def test_revision_turns_with_tools(self) -> None:
        inst, client, rec = self._run("harness_check_tools")
        self.assertEqual(rec["check_result"], "mismatch")
        self.assertEqual(rec["turns"], 1 + EXP2_REVISION_TURNS)
        self.assertEqual(client.tool_choices, [None, None, None, {"type": "none"}])
        self.assertEqual(rec["pred_support"], list(inst.true_support))
        # The check queries are visible to the agent's code but are not charged to its budget.
        self.assertEqual(rec["queries_before_check"], 0)
        self.assertEqual(rec["tool_log"][0]["stdout"].strip(), str(EXP2_CHECK_K))
        self.assertEqual(rec["queries_used"], EXP2_CHECK_K + 5)
        self.assertEqual(rec["tool_sequence"], ["harness_check:revise", "run_python:ok", "query_oracle_random:ok"])

    def test_correct_first_answer_passes_without_revision(self) -> None:
        inst = generate_blackwell_inv_instance(n=40, support_size=3, variant="wb_inv_code", seed=5001)
        right = {"support": list(inst.true_support), "signs": list(inst.true_signs)}
        client = ScriptedClient(lambda i, call: text_reply(right))
        rec = run_exp2_trial(client=client, model="scripted", variant="wb_inv_code", mode="WB",
                             budget_track="matched_medium_budget", seed=5001, protocol="harness_check_tools")
        self.assertEqual((rec["check_result"], rec["turns"], len(client.calls)), ("pass", 1, 1))
        self.assertEqual(rec["exact_rule"], 1.0)


class TestProtocolBudgets(unittest.TestCase):
    """Each protocol's turns, tool permissions and query budget under a client that always wants more queries."""

    def _run(self, protocol: str, mode: str = "WB", track: str = "matched_medium_budget"):
        client = ScriptedClient(greedy_tool_policy)
        rec = run_exp2_trial(client=client, model="scripted", variant="wb_inv_code", mode=mode,
                             budget_track=track, seed=5002, protocol=protocol)
        return client, rec

    def test_standard(self) -> None:
        client, rec = self._run("standard")
        self.assertEqual(client.tool_choices, [None] * 4 + [NO_TOOLS])
        self.assertEqual((rec["turns"], rec["queries_used"], rec["q_max"]), (5, 80, 80))
        self.assertEqual(rec["tool_sequence"], ["query_oracle_random:ok"] * 2 + ["query_oracle_random:err"] * 2)
        self.assertIn("total budget 80 queries", client.calls[0]["messages"][0]["content"])
        self.assertNotIn("protocol", rec)

    def test_verify_first(self) -> None:
        client, rec = self._run("verify_first")
        self.assertEqual(client.tool_choices, [None] * 4 + [NO_TOOLS])
        self.assertEqual((rec["turns"], rec["queries_used"], rec["protocol"]), (5, 80, "verify_first"))
        self.assertIn(VERIFY_FIRST_NOTE, client.calls[0]["messages"][0]["content"])

    def test_harness_check(self) -> None:
        client, rec = self._run("harness_check")
        # One turn fewer for the agent, then one tool-free revision: same total turns as standard.
        self.assertEqual(client.tool_choices, [None] * 3 + [NO_TOOLS, NO_TOOLS])
        self.assertEqual(rec["turns"], 5)
        self.assertEqual(rec["queries_used"], 80 - EXP2_CHECK_K)
        self.assertIn(f"total budget {80 - EXP2_CHECK_K} queries", client.calls[0]["messages"][0]["content"])
        self.assertEqual(rec["check_result"], "mismatch")
        self.assertEqual(rec["tool_sequence"][-1], "harness_check:revise")
        for key in CHECK_FIELDS:
            self.assertIn(key, rec)

    def test_harness_check_tools(self) -> None:
        client, rec = self._run("harness_check_tools")
        self.assertEqual(client.tool_choices, [None] * 4 + [NO_TOOLS] + [None] * (EXP2_REVISION_TURNS - 1) + [NO_TOOLS])
        self.assertEqual(rec["turns"], 5 + EXP2_REVISION_TURNS)
        self.assertEqual(rec["queries_before_check"], 80 - EXP2_CHECK_K)
        self.assertEqual(rec["queries_used"], 80)  # the check queries are added, the agent gets no more
        self.assertEqual(rec["revision_turns"], EXP2_REVISION_TURNS)
        revision_prompt = client.calls[5]["messages"][-1]["content"]
        self.assertIn("appended to `queries.json`", revision_prompt[-1]["text"])

    def test_budget_tracks_set_turns_per_mode(self) -> None:
        self.assertEqual(BUDGET_TRACK_TURNS["matched_query_free_code"], (4, 8))
        for track, (bb_turns, wb_turns) in BUDGET_TRACK_TURNS.items():
            self.assertEqual(turns_for(track, "BB"), bb_turns)
            self.assertEqual(turns_for(track, "WB"), wb_turns)
        with self.assertRaises(ValueError):
            turns_for("no_such_track", "BB")
        client, rec = self._run("standard", mode="BB", track="matched_query_free_code")
        self.assertEqual((len(client.calls), rec["turns"]), (4, 4))
        client, rec = self._run("standard", mode="WB", track="matched_query_free_code")
        self.assertEqual((len(client.calls), rec["turns"]), (8, 8))

    def test_unknown_protocol_and_variant_are_rejected(self) -> None:
        client = ScriptedClient(greedy_tool_policy)
        with self.assertRaises(ValueError):
            run_exp2_trial(client=client, model="m", variant="wb_clean", mode="WB",
                           budget_track="matched_medium_budget", seed=1, protocol="nope")


class TestAgentLoop(unittest.TestCase):
    def _loop(self, policy, *, allow_python: bool = True, max_turns: int = 3, final_check=None, revision_turns: int = 1):
        inst = generate_blackwell_inv_instance(n=40, support_size=3, variant="wb_clean", seed=7)
        client = ScriptedClient(policy)
        with IsolatedPythonSandbox(timeout_s=5.0) as sbx:
            oracle = MeteredOracle(inst, max_queries=20, queries_json_path=sbx.queries_json_path)
            res = run_agent_loop(
                client=client, model="m", system="s", user_prompt="task", oracle=oracle,
                sandbox=sbx if allow_python else None, allow_python=allow_python, max_turns=max_turns,
                max_tokens_per_turn=64, final_check=final_check, revision_turns=revision_turns,
            )
        return client, res

    def test_last_turn_forbids_tools_and_says_so(self) -> None:
        client, res = self._loop(greedy_tool_policy)
        self.assertEqual(client.tool_choices, [None, None, NO_TOOLS])
        last_user = client.calls[-1]["messages"][-1]
        self.assertEqual(last_user["content"][-1], {"type": "text", "text": FINAL_TURN_NOTE})
        self.assertEqual(res["queries_used"], 20)
        self.assertEqual(json.loads(res["final_text"]), {"support": [0, 1, 2], "signs": [1, 1, 1]})

    def test_same_tool_list_on_every_turn(self) -> None:
        client, _ = self._loop(greedy_tool_policy)
        self.assertTrue(all(c["tools"] == client.calls[0]["tools"] for c in client.calls))
        self.assertEqual([t["name"] for t in client.calls[0]["tools"]], ["query_oracle_random", "query_oracle", "run_python"])

    def test_tool_free_track_offers_no_python(self) -> None:
        self.assertEqual([t["name"] for t in build_tool_schemas(False, 40)], ["query_oracle_random", "query_oracle"])
        client, res = self._loop(lambda i, c: tool_reply("run_python", {"code": "print(1)"}) if i == 0 else text_reply("x"),
                                 allow_python=False)
        self.assertEqual(res["tool_sequence"], ["run_python:err"])
        self.assertEqual(res["python_calls"], 0)
        result = json.loads(client.calls[1]["messages"][-1]["content"][0]["content"])
        self.assertEqual(result, {"ok": False, "error": "Tool 'run_python' is unavailable."})

    def test_usage_is_summed_over_turns(self) -> None:
        def policy(i: int, call: Dict[str, Any]) -> Dict[str, Any]:
            usage = {"input_tokens": 10, "output_tokens": 3, "output_tokens_details": {"thinking_tokens": 2}}
            if i < 2:
                return {**tool_reply("query_oracle", {"inputs": [[1] * 40]}, call_id=f"c{i}"), "usage": usage}
            return {**text_reply("done", stop_reason="max_tokens"), "usage": usage}

        _, res = self._loop(policy)
        self.assertEqual((res["input_tokens"], res["output_tokens"], res["thinking_tokens"]), (30, 9, 6))
        self.assertEqual((res["truncated_turns"], res["stop_reasons"]), (1, ["tool_use", "tool_use", "max_tokens"]))
        self.assertEqual(res["queries_used"], 2)

    def test_check_after_a_cut_turn_answers_its_pending_tool_calls(self) -> None:
        def policy(i: int, call: Dict[str, Any]) -> Dict[str, Any]:
            if i == 0:  # a tool call cut at the token cap: the loop stops without executing it
                return {**tool_reply("query_oracle_random", {"count": 1}, call_id="cut"), "stop_reason": "max_tokens"}
            return text_reply("revised")

        client, res = self._loop(policy, final_check=lambda text: "please revise")
        revision_turn = client.calls[1]["messages"][-1]
        self.assertEqual(revision_turn["role"], "user")
        self.assertEqual(revision_turn["content"][0]["type"], "tool_result")
        self.assertEqual(revision_turn["content"][0]["tool_use_id"], "cut")
        self.assertEqual(revision_turn["content"][-1], {"type": "text", "text": "please revise"})
        self.assertEqual((res["harness_check"], res["final_text"], res["queries_used"]), ("revise", "revised", 0))

    def test_spend_cap_interrupts_a_trial(self) -> None:
        client = ScriptedClient(greedy_tool_policy, cost_per_call=1.0, max_cost_usd=2.5)
        with self.assertRaises(BudgetExceededError):
            run_exp2_trial(client=client, model="m", variant="wb_clean", mode="WB",
                           budget_track="matched_medium_budget", seed=3)
        self.assertEqual(len(client.calls), 3)


class TestRecordSchema(unittest.TestCase):
    def test_exp2_record_fields(self) -> None:
        rec = run_exp2_trial(client=ScriptedClient(greedy_tool_policy), model="m", variant="wb_inv_relu", mode="WB",
                             budget_track="matched_total_budget", seed=5003)
        self.assertEqual(list(rec), EXP2_FIELDS)
        rec = run_exp2_trial(client=ScriptedClient(greedy_tool_policy), model="m", variant="wb_clean", mode="WB",
                             budget_track="matched_total_budget", seed=5003, protocol="harness_check_tools",
                             log_tool_inputs=True)
        extra = ["protocol", "revision_turns", "queries_before_check", *CHECK_FIELDS, "tool_log", "oracle_history"]
        self.assertEqual(list(rec), extra + EXP2_FIELDS)
        self.assertTrue(all(set(h["x"]) <= {"0", "1"} and len(h["x"]) == 40 for h in rec["oracle_history"]))

    def test_exp1_record_fields_and_budget(self) -> None:
        for track, turns in (("tool_free", 2), ("tool_python", 6)):
            client = ScriptedClient(greedy_tool_policy)
            rec = run_exp1_trial(client=client, model="m", family="4xor_n40", regime="vbb_hard", substrate="code_clean",
                                 mode="WB", track=track, seed=1001, planted=1)
            self.assertEqual(list(rec), EXP1_FIELDS)
            self.assertEqual((len(client.calls), rec["queries_used"]), (turns, 40))
            self.assertEqual(rec["decision_label"], "PARSE_ERROR")
            self.assertEqual(rec["correct"], 0)

    def test_unknown_exp1_family_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_exp1_trial(client=ScriptedClient(greedy_tool_policy), model="m", family="5xor", regime="vbb_hard",
                           substrate="code_clean", mode="BB", track="tool_free", seed=1, planted=0)


if __name__ == "__main__":
    unittest.main()
