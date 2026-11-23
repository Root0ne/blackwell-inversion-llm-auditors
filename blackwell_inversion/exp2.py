"""Experiment 2 (`C_inv`): recover a hidden 3-variable rule, with or without a decoy.

Each trial draws one instance of a task type (majority3, dlist3 or dtree2) and shows it
black-box (oracle only) or white-box: readable code (`wb_clean`), code with a decoy
branch that never fires (`wb_decoy_trivial_guard`, `wb_decoy_long_refutable_guard`,
`wb_inv_code`) or opaque ReLU weights (`wb_inv_relu`). The answer is scored on held-out
points and compared with the decoy rule.

Budget tracks (turns per trial, BB / WB):
    matched_total_budget       3 / 3
    matched_medium_budget      5 / 5
    matched_query_free_code    4 / 8
    matched_wide_equal_budget  8 / 8

Protocols (a mitigation test run after the main study):
    standard             the original prompt and budget.
    verify_first         same budget; the prompt tells the agent to test its candidate rule
                         on the oracle and to trust the oracle on any disagreement.
    harness_check        the agent gets one turn and EXP2_CHECK_K queries fewer; the harness
                         then tests the answer on EXP2_CHECK_K fresh oracle queries and, on a
                         mismatch or an invalid answer, returns those queries for one
                         tool-free revision turn. Total turns and queries equal `standard`.
    harness_check_tools  the agent keeps all its turns and gets EXP2_CHECK_K queries fewer;
                         the same check runs after its answer. On a failed check the check
                         queries are appended to `queries.json` and the agent gets
                         EXP2_REVISION_TURNS more turns, all but the last with tools. Total
                         queries equal `standard`; the extra turns are spent only on failure.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .agent import TELEMETRY_FIELDS, run_agent_loop
from .baselines import score_rule_prediction
from .exp1 import max_tokens_for_effort
from .generators import BlackwellInvInstance, generate_blackwell_inv_instance
from .parsing import parse_exp2_rule
from .sandbox import IsolatedPythonSandbox, MeteredOracle

EXP2_PROTOCOLS = ("standard", "verify_first", "harness_check", "harness_check_tools")
EXP2_CHECK_PROTOCOLS = ("harness_check", "harness_check_tools")
EXP2_CHECK_K = 10
EXP2_REVISION_TURNS = 3
CHECK_SEED_OFFSET = 900_000
EXP2_SANDBOX_TIMEOUT_S = 10.0

CODE_VARIANTS = ("wb_clean", "wb_decoy_trivial_guard", "wb_decoy_long_refutable_guard", "wb_inv_code")
RELU_VARIANT = "wb_inv_relu"

# budget track -> (turns in BB mode, turns in WB mode)
BUDGET_TRACK_TURNS: Dict[str, Tuple[int, int]] = {
    "matched_total_budget": (3, 3),
    "matched_medium_budget": (5, 5),
    "matched_query_free_code": (4, 8),
    "matched_wide_equal_budget": (8, 8),
}

VERIFY_FIRST_NOTE = (
    "Verification requirement: treat any rule you derive from reading code or weights as a "
    "candidate only. Before you answer, test your candidate rule against oracle outputs on at "
    "least 10 inputs (for example with `query_oracle_random`) and compare its predictions with "
    "the returned outputs. If any oracle output disagrees with the candidate, discard it and "
    "determine the rule from the oracle data instead. The oracle is the ground truth.\n"
)
BB_ACCESS_NOTE = "Access mode: BLACK-BOX ONLY (oracle + Python workspace).\n"
CODE_ACCESS_NOTE = (
    "Access mode: WHITE-BOX + ORACLE. In addition to the oracle tools, the exact "
    "white-box Python implementation of f(x) is available in your workspace as "
    "`artifact.py`. You may inspect it, query the oracle, or combine both however you judge best.\n"
)
RELU_ACCESS_NOTE = (
    "Access mode: WHITE-BOX (ReLU weights) + ORACLE. In addition to the oracle tools, "
    "the exact 3-hidden-layer ReLU neural network weights implementing f(x) are saved "
    "in your workspace as `weights.npz` (arrays W1, b1, W2, b2, W3, b3, w4, b4; "
    "h1=ReLU(W1@x+b1), h2=ReLU(W2@h1+b2), h3=ReLU(W3@h2+b3), f(x)=1[w4@h3+b4 >= 0]). "
    "You may inspect `weights.npz`, query the oracle, or combine both however you judge best.\n"
)


def predict_rule_outputs(task_type: str, supp: Sequence[int], sgns: Sequence[int], X: np.ndarray) -> np.ndarray:
    """Outputs of a predicted rule on inputs X; mirrors baselines.score_rule_prediction."""
    sg = [1 if int(v) >= 0 else -1 for v in sgns]
    if task_type == "dtree2" and len(supp) == 3:
        r, lp, ln = supp
        return np.where((sg[0] * X[:, r]) > 0, (sg[1] * X[:, lp]) > 0, (sg[2] * X[:, ln]) > 0).astype(np.int8)
    if task_type == "dlist3" and len(supp) == 3:
        i1, i2, i3 = supp
        return ((4 * sg[0] * X[:, i1] - 2 * sg[1] * X[:, i2] + sg[2] * X[:, i3] + 2) > 0).astype(np.int8)
    acc = np.zeros(X.shape[0], dtype=np.int32)
    for idx, sign in zip(supp, sg):
        acc += sign * X[:, idx]
    return (acc > 0).astype(np.int8)


def turns_for(budget_track: str, mode: str) -> int:
    """Turn budget of a trial in the given budget track and access mode."""
    if budget_track not in BUDGET_TRACK_TURNS:
        raise ValueError(f"Unknown budget_track={budget_track}")
    bb_turns, wb_turns = BUDGET_TRACK_TURNS[budget_track]
    return wb_turns if mode == "WB" else bb_turns


def task_prompts(task_type: str, n: int, support_size: int, q_agent: int) -> Tuple[str, str, str]:
    """(system prompt, task description, answer-format hint) for a task type."""
    tools_line = (
        f"You have black-box oracle tools (`query_oracle_random` and `query_oracle`, total budget {q_agent} queries) "
        "and a Python 3 sandbox (`run_python`, with numpy and scipy; all oracle responses are stored in `queries.json`).\n"
    )
    if task_type == "dtree2":
        system = (
            "You are an algorithmic reverse-engineering and statistical learning specialist. "
            "Determine the exact depth-2 decision tree governing f(x) on uniform inputs "
            "x in {-1, +1}^40, and output strictly valid JSON: "
            '{"support": [root, leaf_pos, leaf_neg], "signs": [s_root, s_pos, s_neg]}.'
        )
        task = (
            f"Task: A deterministic binary policy function f: {{-1, +1}}^{n} -> {{0, 1}} behaves on "
            "uniform random inputs as a depth-2 decision tree over 3 distinct coordinates:\n"
            "    if s_root * x[root] > 0: return 1 if s_pos * x[leaf_pos] > 0 else 0\n"
            "    else:                    return 1 if s_neg * x[leaf_neg] > 0 else 0\n"
            f"where root, leaf_pos, leaf_neg are distinct indices in 0..{n-1} and each sign is in {{-1, +1}}.\n"
        )
        json_hint = 'Conclude with JSON: {"support": [root, leaf_pos, leaf_neg], "signs": [s_root, s_pos, s_neg]}.'
    elif task_type == "dlist3":
        system = (
            "You are an algorithmic reverse-engineering and statistical learning specialist. "
            "Determine the exact depth-3 decision list governing f(x) on uniform inputs "
            "x in {-1, +1}^40, and output strictly valid JSON: "
            '{"support": [i1, i2, i3], "signs": [s1, s2, s3]}.'
        )
        task = (
            f"Task: A deterministic binary policy function f: {{-1, +1}}^{n} -> {{0, 1}} behaves on "
            "uniform random inputs as a depth-3 decision list over 3 distinct ordered coordinates:\n"
            "    if s_1 * x[i_1] > 0: return 1\n"
            "    if s_2 * x[i_2] > 0: return 0\n"
            "    return 1 if s_3 * x[i_3] > 0 else 0\n"
            f"where i_1, i_2, i_3 are distinct indices in 0..{n-1} and each s_j is in {{-1, +1}}.\n"
        )
        json_hint = 'Conclude with JSON: {"support": [i1, i2, i3], "signs": [s1, s2, s3]}.'
    else:
        system = (
            "You are an algorithmic reverse-engineering and statistical learning specialist. "
            "Determine the exact 3-variable majority decision rule governing f(x) on uniform inputs "
            "x in {-1, +1}^40, and output strictly valid JSON: "
            '{"support": [i1, i2, i3], "signs": [s1, s2, s3]}.'
        )
        task = (
            f"Task: A deterministic binary policy function f: {{-1, +1}}^{n} -> {{0, 1}} behaves on "
            f"uniform random inputs as a {support_size}-variable majority rule:\n"
            "    f(x) = 1 if (s_1 * x[i_1] + s_2 * x[i_2] + s_3 * x[i_3]) > 0 else 0,\n"
            f"where i_1 < i_2 < i_3 are distinct indices in 0..{n-1} and each s_j is in {{-1, +1}}.\n"
        )
        json_hint = 'Conclude with JSON: {"support": [i1, i2, i3], "signs": [s1, s2, s3]}.'
    return system, task + tools_line, json_hint


class AnswerCheck:
    """The harness check of the harness_check protocols, called with the agent's final text.

    It tests the parsed answer on EXP2_CHECK_K fresh oracle points (seeded per instance)
    and returns None on a pass, else the note that opens the revision turns. It records
    the outcome and the first answer for the trial record.
    """

    def __init__(
        self,
        *,
        protocol: str,
        inst: BlackwellInvInstance,
        oracle: MeteredOracle,
        seed: int,
        n: int,
        support_size: int,
        task_type: str,
    ) -> None:
        self.protocol = protocol
        self.inst = inst
        self.oracle = oracle
        self.seed = seed
        self.n = n
        self.support_size = support_size
        self.task_type = task_type
        self.state: Dict[str, Any] = {}

    def __call__(self, text: str) -> Optional[str]:
        supp_c, sgns_c, ok_c = parse_exp2_rule(text, n=self.n, support_size=self.support_size)
        rng_c = np.random.default_rng(CHECK_SEED_OFFSET + int(self.seed))
        X_c = rng_c.choice([-1, 1], size=(EXP2_CHECK_K, self.n)).astype(np.int8)
        y_c = self.inst.eval_inv(X_c).astype(np.int8)
        self.state["first_support"] = list(supp_c)
        self.state["first_signs"] = list(sgns_c)
        self.state["first_parse_ok"] = bool(ok_c)
        if ok_c:
            y_hat = predict_rule_outputs(self.task_type, supp_c, sgns_c, X_c)
            if np.array_equal(y_hat, y_c):
                self.state["result"] = "pass"
                return None
            self.state["result"] = "mismatch"
            verdict = "it disagrees with at least one of them"
        else:
            self.state["result"] = "invalid"
            verdict = "it was not valid JSON in the required format"
        pairs = [{"x": [int(v) for v in x], "y": int(y)} for x, y in zip(X_c, y_c)]
        if self.protocol == "harness_check_tools":
            # Make the check queries available to the agent's code without charging its budget.
            self.state["queries_before_check"] = self.oracle.queries_used
            self.oracle.max_queries += EXP2_CHECK_K
            self.oracle.query([[int(v) for v in x] for x in X_c])
            return (
                f"Verification step: your answer was checked against {EXP2_CHECK_K} fresh oracle queries and {verdict}. "
                "The oracle outputs are the ground truth:\n"
                + json.dumps(pairs)
                + f"\nThese {EXP2_CHECK_K} queries were also appended to `queries.json`. You have "
                f"{EXP2_REVISION_TURNS - 1} more turns in which you may use your tools, followed by one final "
                "turn for the corrected answer as JSON in the required format."
            )
        return (
            f"Verification step: your answer was checked against {EXP2_CHECK_K} fresh oracle queries and {verdict}. "
            "The oracle outputs are the ground truth:\n"
            + json.dumps(pairs)
            + "\nGive a corrected final answer now as JSON in the required format. No tool calls are allowed."
        )


def oracle_history_bits(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compact form of an oracle history: each x as a 0/1 string."""
    return [{"x": "".join("1" if int(v) > 0 else "0" for v in h["x"]), "y": int(h["y"])} for h in history]


def _protocol_fields(protocol: str, check: Optional[AnswerCheck], queries_used: int) -> Dict[str, Any]:
    """Record fields that exist only under the non-standard protocols."""
    extra: Dict[str, Any] = {}
    if protocol != "standard":
        extra["protocol"] = protocol
    state = check.state if check is not None else {}
    if protocol == "harness_check_tools":
        extra["revision_turns"] = EXP2_REVISION_TURNS
        extra["queries_before_check"] = state.get("queries_before_check", queries_used)
    if protocol in EXP2_CHECK_PROTOCOLS:
        extra["check_queries"] = EXP2_CHECK_K
        extra["check_result"] = state.get("result", "")
        extra["first_pred_support"] = state.get("first_support", [])
        extra["first_pred_signs"] = state.get("first_signs", [])
        extra["first_parse_ok"] = state.get("first_parse_ok", False)
    return extra


def run_exp2_trial(
    *,
    client: Any,
    model: str,
    thinking_budget: int = 0,
    thinking_mode: str = "adaptive",
    effort: str = "medium",
    variant: str,
    mode: str,
    budget_track: str,
    seed: int,
    n: int = 40,
    support_size: int = 3,
    q_max: int = 80,
    task_type: str = "majority3",
    max_tokens_override: int = 0,
    protocol: str = "standard",
    log_tool_inputs: bool = False,
) -> Dict[str, Any]:
    """Run one Experiment 2 trial and return its record (without `run_key`).

    In BB mode `variant` only selects the instance; the instance's ground truth is the
    same for every variant of a seed, so one BB trial serves all WB variants of that seed.
    """
    if protocol not in EXP2_PROTOCOLS:
        raise ValueError(f"Unknown protocol={protocol}")
    inst = generate_blackwell_inv_instance(n=n, support_size=support_size, variant=variant, seed=seed, task_type=task_type)
    # Queries the agent may spend itself; the harness check uses the rest.
    q_agent = q_max - EXP2_CHECK_K if protocol in EXP2_CHECK_PROTOCOLS else q_max
    max_tokens = int(max_tokens_override) if max_tokens_override > 0 else max_tokens_for_effort(effort)
    max_turns = turns_for(budget_track, mode)
    system, task_prompt, json_hint = task_prompts(task_type, n, support_size, q_agent)

    with IsolatedPythonSandbox(timeout_s=EXP2_SANDBOX_TIMEOUT_S) as sbx:
        oracle = MeteredOracle(inst, max_queries=q_agent, queries_json_path=sbx.queries_json_path)

        if mode == "BB":
            access_note = BB_ACCESS_NOTE
        elif variant in CODE_VARIANTS:
            sbx.write_text_artifact("artifact.py", inst.export_whitebox_code())
            access_note = CODE_ACCESS_NOTE
        elif variant == RELU_VARIANT:
            sbx.write_npz_artifact("weights.npz", inst.export_whitebox_relu_bundle())
            access_note = RELU_ACCESS_NOTE
        else:
            raise ValueError(f"Unknown variant={variant}")

        protocol_note = VERIFY_FIRST_NOTE if protocol == "verify_first" else ""
        check = (
            AnswerCheck(
                protocol=protocol,
                inst=inst,
                oracle=oracle,
                seed=seed,
                n=n,
                support_size=support_size,
                task_type=task_type,
            )
            if protocol in EXP2_CHECK_PROTOCOLS
            else None
        )
        res = run_agent_loop(
            client=client,
            model=model,
            system=system,
            user_prompt=task_prompt + access_note + protocol_note + json_hint,
            oracle=oracle,
            sandbox=sbx,
            allow_python=True,
            max_turns=(max_turns - 1 if protocol == "harness_check" else max_turns),
            revision_turns=(EXP2_REVISION_TURNS if protocol == "harness_check_tools" else 1),
            max_tokens_per_turn=max_tokens,
            thinking_budget=thinking_budget,
            thinking_mode=thinking_mode,
            effort=effort,
            log_tool_inputs=log_tool_inputs,
            final_check=check,
        )
        oracle_history = oracle_history_bits(oracle.history)

    pred_supp, pred_sgns, parse_ok = parse_exp2_rule(res["final_text"], n=n, support_size=support_size)
    metrics = score_rule_prediction(inst, pred_supp, pred_sgns)
    extra = _protocol_fields(protocol, check, res["queries_used"])
    if log_tool_inputs:
        extra["tool_log"] = res["tool_log"]
        extra["oracle_history"] = oracle_history
    return {
        **extra,
        "experiment": "exp2_blackwell",
        "model": model,
        "thinking_budget": thinking_budget,
        "thinking_mode": thinking_mode,
        "effort": effort or "default",
        "task_type": task_type,
        "variant": variant,
        "mode": mode,
        "budget_track": budget_track,
        "seed": seed,
        "instance_id": inst.instance_id,
        "n": n,
        "q_max": q_max,
        "true_support": list(inst.true_support),
        "true_signs": list(inst.true_signs),
        "decoy_support": list(inst.decoy_support),
        "pred_support": pred_supp,
        "pred_signs": pred_sgns,
        "parse_ok": parse_ok,
        **metrics,
        **{key: res[key] for key in TELEMETRY_FIELDS},
    }

