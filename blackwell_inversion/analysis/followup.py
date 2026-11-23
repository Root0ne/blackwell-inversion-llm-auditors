"""Analysis of the follow-up runs made after the main benchmark was locked.

These runs are NOT part of the 7,631 trials of the main data set. They reuse the
main condition of Experiment 2 (Majority-3, 5 turns BB / 5 turns WB, medium effort,
seeds 5000-5024) and are stored in two separate files:

  results_c4_refutable.jsonl  Part A. The refutable control run on the models that
                              lacked it in the main benchmark.
  results_c4_protocols.jsonl  Part B. Three protocols (standard, verify_first,
                              harness_check) x four conditions (BB, clean code,
                              hard-guard code, refutable control) on four models.

Part A pairs each new refutable-control trial with the hard-guard-code trial and the
BB trial of the same model and seed in the locked data, and repeats the pooled
equivalence analysis of the mechanism module on the extended set of cells.

Part B compares the protocols within model and seed. It also compares the freshly run
standard protocol with the locked records of the same cells (replication check).

Part D reads results_c5_check_tools.jsonl: the answer check repeated with all of the
agent's turns kept and, after a failed check, three more turns of which two may use tools
(protocol harness_check_tools; BB and hard-guard code, the four models of Part B, 25 seeds).
It is compared by seed with the standard and harness_check protocols of Part B and, for the
two Claude models, set beside the locked 8-turn condition.

Part C is a text analysis prompted by the follow-up logs: it counts how many answers that
state the decoy rule are presented by the agent itself as a guess or as unverified. It
uses the first 240 characters of the final message (`final_excerpt`), which every record
of the locked data also holds, and a fixed keyword pattern (HEDGE). The same pattern is
applied to exactly correct answers to show how often it fires without a decoy answer.

Scoring convention as in the other analysis scripts: an answer that is not a valid
rule scores heldout_acc = 0.50 and matches_decoy = 0.

Run from the repository root:
    python -m blackwell_inversion.analysis.followup --out-json reports/report_followup.json \
        > reports/report_followup.txt
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..baselines import score_rule_prediction
from ..exp2 import predict_rule_outputs
from ..paths import CHECK_TOOLS_FILE, PROTOCOLS_FILE, REFUTABLE_FILE
from .mechanism import (
    CLEAN,
    CODE,
    N_BOOT,
    N_PERM,
    REFUT,
    SEED,
    acc,
    arm,
    both_valid,
    build_pairs,
    cell_key,
    contrast,
    instance_for,
    is_main,
    load_exp2,
    mt_override,
    pooled_paired,
    strat_two_group,
    trapped,
)
from .stats import load_jsonl, paired_delta_stats

PROTOCOLS = ("standard", "verify_first", "harness_check")
ARMS_B = ("BB", CLEAN, CODE, REFUT)
Record = Dict[str, Any]


def proto(r: Record) -> str:
    return r.get("protocol", "standard")


def paired(xa: Sequence[float], xb: Sequence[float]) -> Dict[str, Any]:
    st = paired_delta_stats(xa, xb, alternative="two-sided", n_perm=N_PERM, n_boot=N_BOOT, seed=SEED)
    return {
        "n_pairs": st["n_pairs"],
        "mean_a": st["mean_wb"],
        "mean_b": st["mean_bb"],
        "diff": st["delta_wb"],
        "ci95": [st["ci_lo"], st["ci_hi"]],
        "p_signflip": st["p_value"],
    }


def wilson(k: int, n: int, z: float = 1.959964) -> List[float]:
    if n == 0:
        return [0.0, 0.0]
    ph = k / n
    den = 1 + z * z / n
    mid = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [float(max(0.0, mid - half)), float(min(1.0, mid + half))]


def boot_mean_ci(x: Sequence[float], seed: int = SEED) -> List[float]:
    a = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    m = a[rng.integers(0, len(a), size=(N_BOOT, len(a)))].mean(axis=1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def describe(rs: Sequence[Record]) -> Dict[str, Any]:
    n = len(rs)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "acc": float(np.mean([acc(r) for r in rs])),
        "valid_json": float(np.mean([1.0 if r.get("parse_ok") else 0.0 for r in rs])),
        "trap_n": int(sum(trapped(r) for r in rs)),
        "trap_rate": float(np.mean([trapped(r) for r in rs])),
        "exact_rule": float(np.mean([float(r.get("exact_rule", 0.0)) for r in rs])),
        "mean_queries": float(np.mean([r["queries_used"] for r in rs])),
        "zero_query_share": float(np.mean([1.0 if r["queries_used"] == 0 else 0.0 for r in rs])),
        "mean_turns": float(np.mean([r["turns"] for r in rs])),
        "truncated_trials": int(sum(1 for r in rs if r.get("truncated_turns", 0) > 0)),
        "cost_usd": float(sum(r["cost_usd"] for r in rs)),
    }


# ==============================================================================
# EVIDENCE HELD BY THE AGENT -- does its own query log contradict its answer?
# ==============================================================================
def own_query_evidence(r: Record) -> Optional[Dict[str, int]]:
    """Compare the final answer with the agent's own oracle queries (logged as oracle_history).

    Returns the number of queries and the number of them on which the answered rule
    disagrees with the oracle output; None if the answer is invalid or no log was kept."""
    if "oracle_history" not in r or not r.get("parse_ok"):
        return None
    hist = r["oracle_history"]
    if not hist:
        return {"n_queries": 0, "n_contradicted": 0}
    X = np.array([[1 if ch == "1" else -1 for ch in h["x"]] for h in hist], dtype=np.int8)
    y = np.array([int(h["y"]) for h in hist], dtype=np.int8)
    y_hat = predict_rule_outputs(r["task_type"], r["pred_support"], r["pred_signs"], X)
    return {"n_queries": int(len(hist)), "n_contradicted": int(np.sum(y_hat != y))}


def evidence_summary(rs: Sequence[Record]) -> Dict[str, Any]:
    """Summary for trapped answers and for exactly correct answers (sanity: the latter must show 0)."""
    out: Dict[str, Any] = {}
    for name, sel in (("trapped", lambda r: trapped(r) >= 1.0), ("exact", lambda r: float(r.get("exact_rule", 0.0)) >= 1.0)):
        ev = [e for e in (own_query_evidence(r) for r in rs if sel(r)) if e is not None]
        nq = [e["n_queries"] for e in ev]
        nc = [e["n_contradicted"] for e in ev]
        out[name] = {
            "n": len(ev),
            "queried": int(sum(1 for q in nq if q > 0)),
            "queried_ge10": int(sum(1 for q in nq if q >= 10)),
            "contradicted_by_own_queries": int(sum(1 for c in nc if c > 0)),
            "mean_queries": float(np.mean(nq)) if nq else None,
            "mean_contradicting_queries": float(np.mean(nc)) if nc else None,
            "median_contradicting_queries_among_queried": float(np.median([c for q, c in zip(nq, nc) if q > 0])) if any(q > 0 for q in nq) else None,
        }
    return out


# ==============================================================================
# PART A -- refutable control on the models that lacked it
# ==============================================================================
def part_a(locked: Sequence[Record], new: Sequence[Record]) -> Dict[str, Any]:
    res: Dict[str, Any] = {"n_new_trials": len(new), "cost_usd": float(sum(r["cost_usd"] for r in new))}
    main_locked = [r for r in locked if is_main(r)]
    by_model: Dict[str, Dict[str, Dict[int, Record]]] = defaultdict(lambda: defaultdict(dict))
    for r in main_locked:
        by_model[r["model"]][arm(r)][int(r["seed"])] = r
    new_by_model: Dict[str, Dict[int, Record]] = defaultdict(dict)
    for r in new:
        assert is_main(r) and r["variant"] == REFUT and r["mode"] == "WB", r["run_key"]
        assert int(r["seed"]) not in new_by_model[r["model"]], f"duplicate {r['run_key']}"
        assert REFUT not in by_model[r["model"]], f"{r['model']} already has the refutable control"
        new_by_model[r["model"]][int(r["seed"])] = r

    models_out = []
    for m in sorted(new_by_model):
        nr = new_by_model[m]
        seeds = sorted(nr)
        code, bb = by_model[m][CODE], by_model[m]["BB"]
        common = [s for s in seeds if s in code and s in bb]
        row = {
            "model": m,
            "refutable": describe([nr[s] for s in seeds]),
            "hard_guard_locked": describe([code[s] for s in common]),
            "bb_locked": describe([bb[s] for s in common]),
            "delta_wb_refutable": paired([acc(nr[s]) for s in common], [acc(bb[s]) for s in common]),
            "acc_code_minus_refut": paired([acc(code[s]) for s in common], [acc(nr[s]) for s in common]),
            "trap_code_minus_refut": paired([trapped(code[s]) for s in common], [trapped(nr[s]) for s in common]),
        }
        models_out.append(row)
    res["models"] = models_out

    # Pooled equivalence on the extended set of cells (locked cells + new cells).
    by = build_pairs(list(locked) + list(new))
    for name, metric, seed in (("acc", acc, SEED), ("trap", trapped, SEED + 1)):
        cells, diffs = contrast(by, CODE, REFUT, metric)
        res[f"{name}_code_minus_refut_extended"] = {
            "n_models": len({c["cell"][0] for c in cells}),
            "pooled": pooled_paired(diffs, seed=seed),
        }
    cells_v, diffs_v = contrast(by, CODE, REFUT, acc, pair_filter=both_valid)
    res["acc_code_minus_refut_extended_valid_json_only"] = {"pooled": pooled_paired(diffs_v, seed=SEED + 4)}

    # New cells only, and the main condition only (one cell per model).
    new_cells = {(m, r["effort"], r["budget_track"], r["task_type"], 0) for m, d in new_by_model.items() for r in d.values()}
    by_new = {c: v for c, v in by.items() if c in new_cells}
    for name, metric, seed in (("acc", acc, SEED + 5), ("trap", trapped, SEED + 6)):
        cells, diffs = contrast(by_new, CODE, REFUT, metric)
        res[f"{name}_code_minus_refut_new_cells_only"] = {"cells": cells, "pooled": pooled_paired(diffs, seed=seed)}
    by_main = build_pairs([r for r in list(locked) + list(new) if is_main(r)])
    for name, metric, seed in (("acc", acc, SEED + 7), ("trap", trapped, SEED + 8)):
        cells, diffs = contrast(by_main, CODE, REFUT, metric)
        res[f"{name}_code_minus_refut_main_condition"] = {
            "n_models": len({c["cell"][0] for c in cells}),
            "cells": cells,
            "pooled": pooled_paired(diffs, seed=seed),
        }
    # Trap rates of both variants per model in the main condition (for the figure).
    per_model = []
    for c in sorted(by_main):
        if CODE in by_main[c] and REFUT in by_main[c]:
            per_model.append(
                {
                    "model": c[0],
                    "n_code": len(by_main[c][CODE]),
                    "n_refut": len(by_main[c][REFUT]),
                    "trap_code": float(np.mean([trapped(r) for r in by_main[c][CODE].values()])),
                    "trap_refut": float(np.mean([trapped(r) for r in by_main[c][REFUT].values()])),
                    "acc_code": float(np.mean([acc(r) for r in by_main[c][CODE].values()])),
                    "acc_refut": float(np.mean([acc(r) for r in by_main[c][REFUT].values()])),
                    "refut_source": "followup" if c[0] in new_by_model else "locked",
                }
            )
    res["main_condition_per_model"] = per_model
    res["own_query_evidence"] = evidence_summary(list(new))
    return res


# ==============================================================================
# PART B -- protocols
# ==============================================================================
def first_answer_score(r: Record) -> Dict[str, float]:
    """Score of the answer given before the harness check (harness_check protocol only)."""
    inst = instance_for(r)
    return score_rule_prediction(inst, r.get("first_pred_support") or [], r.get("first_pred_signs") or [])


def part_b(locked: Sequence[Record], new: Sequence[Record]) -> Dict[str, Any]:
    res: Dict[str, Any] = {"n_trials": len(new), "cost_usd": float(sum(r["cost_usd"] for r in new))}
    idx: Dict[Tuple[str, str, str], Dict[int, Record]] = defaultdict(dict)
    for r in new:
        assert is_main(r), r["run_key"]
        k = (r["model"], proto(r), arm(r))
        assert int(r["seed"]) not in idx[k], f"duplicate {r['run_key']}"
        idx[k][int(r["seed"])] = r
    models = sorted({k[0] for k in idx})
    res["models"] = models
    res["cost_by_model"] = {m: float(sum(r["cost_usd"] for r in new if r["model"] == m)) for m in models}

    # 1. Descriptives per model x protocol x arm, and Delta_WB within each protocol.
    cells = []
    for m in models:
        for p in PROTOCOLS:
            bb = idx.get((m, p, "BB"), {})
            for a in ARMS_B:
                rs = idx.get((m, p, a), {})
                if not rs:
                    continue
                row = {"model": m, "protocol": p, "arm": a, **describe(list(rs.values()))}
                if a != "BB":
                    common = sorted(set(rs) & set(bb))
                    row["delta_wb"] = paired([acc(rs[s]) for s in common], [acc(bb[s]) for s in common])
                cells.append(row)
    res["cells"] = cells

    # 1b. Per model x protocol summary for the figure and the table: the two decoy-code
    #     variants pooled, with the BB and clean-code accuracy of the same protocol.
    summ = []
    for m in models:
        for p in PROTOCOLS:
            dc = [r for a in (CODE, REFUT) for r in idx.get((m, p, a), {}).values()]
            bb = list(idx.get((m, p, "BB"), {}).values())
            cl = list(idx.get((m, p, CLEAN), {}).values())
            if not dc:
                continue
            k = int(sum(trapped(r) for r in dc))
            summ.append(
                {
                    "model": m,
                    "protocol": p,
                    "n_decoy_code": len(dc),
                    "acc_decoy_code": float(np.mean([acc(r) for r in dc])),
                    "acc_decoy_code_ci95": boot_mean_ci([acc(r) for r in dc]),
                    "trap_n": k,
                    "trap_rate": k / len(dc),
                    "trap_ci95": wilson(k, len(dc)),
                    "valid_json_decoy_code": float(np.mean([1.0 if r.get("parse_ok") else 0.0 for r in dc])),
                    "truncated_decoy_code": int(sum(1 for r in dc if r.get("truncated_turns", 0) > 0)),
                    "mean_queries_decoy_code": float(np.mean([r["queries_used"] for r in dc])),
                    "n_bb": len(bb),
                    "acc_bb": float(np.mean([acc(r) for r in bb])) if bb else None,
                    "acc_bb_ci95": boot_mean_ci([acc(r) for r in bb]) if bb else None,
                    "valid_json_bb": float(np.mean([1.0 if r.get("parse_ok") else 0.0 for r in bb])) if bb else None,
                    "acc_clean": float(np.mean([acc(r) for r in cl])) if cl else None,
                }
            )
    res["summary_by_model_protocol"] = summ

    # 2. Protocol minus standard, per model and arm, and pooled over models (strata = models).
    effects = []
    for p in PROTOCOLS[1:]:
        for a in ARMS_B + ("decoy_code",):
            arms = (CODE, REFUT) if a == "decoy_code" else (a,)
            for metric_name, metric in (("acc", acc), ("trap", trapped)):
                if a in ("BB", CLEAN) and metric_name == "trap":
                    continue
                per_model, diffs = [], []
                for m in models:
                    xa, xb = [], []
                    for aa in arms:
                        A, B = idx.get((m, p, aa), {}), idx.get((m, "standard", aa), {})
                        for s in sorted(set(A) & set(B)):
                            xa.append(metric(A[s]))
                            xb.append(metric(B[s]))
                    if not xa:
                        continue
                    per_model.append({"model": m, **paired(xa, xb)})
                    diffs.append(np.asarray(xa) - np.asarray(xb))
                if diffs:
                    effects.append(
                        {
                            "protocol": p,
                            "arm": a,
                            "metric": metric_name,
                            "per_model": per_model,
                            "pooled": pooled_paired(diffs, seed=SEED + 11),
                        }
                    )
    res["protocol_minus_standard"] = effects

    # 3. Pooled Delta_WB (decoy code vs BB of the same protocol) per protocol.
    dwb = []
    for p in PROTOCOLS:
        for a in (CLEAN, CODE, REFUT, "decoy_code"):
            arms = (CODE, REFUT) if a == "decoy_code" else (a,)
            diffs = []
            for m in models:
                bb = idx.get((m, p, "BB"), {})
                d = []
                for aa in arms:
                    A = idx.get((m, p, aa), {})
                    d += [acc(A[s]) - acc(bb[s]) for s in sorted(set(A) & set(bb))]
                if d:
                    diffs.append(np.asarray(d))
            if diffs:
                dwb.append({"protocol": p, "arm": a, "pooled": pooled_paired(diffs, seed=SEED + 12)})
    res["delta_wb_pooled"] = dwb

    # 4. What the harness check did.
    hc = []
    for m in models + ["ALL"]:
        for a in ARMS_B:
            rs = [
                r
                for (mm, p, aa), d in idx.items()
                if p == "harness_check" and aa == a and (m == "ALL" or mm == m)
                for r in d.values()
            ]
            if not rs:
                continue
            first = [first_answer_score(r) for r in rs]
            flagged = [(r, f) for r, f in zip(rs, first) if r.get("check_result") in ("mismatch", "invalid")]
            first_trap = [(r, f) for r, f in zip(rs, first) if f["matches_decoy"] >= 1.0]
            first_exact = [(r, f) for r, f in zip(rs, first) if f["exact_rule"] >= 1.0]
            hc.append(
                {
                    "model": m,
                    "arm": a,
                    "n": len(rs),
                    "first_answer_acc": float(np.mean([f["heldout_acc"] for f in first])),
                    "final_acc": float(np.mean([acc(r) for r in rs])),
                    "check_pass": int(sum(1 for r in rs if r.get("check_result") == "pass")),
                    "check_mismatch": int(sum(1 for r in rs if r.get("check_result") == "mismatch")),
                    "check_invalid": int(sum(1 for r in rs if r.get("check_result") == "invalid")),
                    "first_trapped": len(first_trap),
                    "first_trapped_flagged": int(sum(1 for r, _ in first_trap if r.get("check_result") == "mismatch")),
                    "first_trapped_final_trapped": int(sum(1 for r, _ in first_trap if trapped(r))),
                    "first_trapped_final_exact": int(sum(1 for r, _ in first_trap if r.get("exact_rule", 0.0) >= 1.0)),
                    "first_exact": len(first_exact),
                    "first_exact_flagged": int(sum(1 for r, _ in first_exact if r.get("check_result") != "pass")),
                    "flagged": len(flagged),
                    "flagged_final_exact": int(sum(1 for r, _ in flagged if r.get("exact_rule", 0.0) >= 1.0)),
                    "flagged_mean_acc_before": float(np.mean([f["heldout_acc"] for _, f in flagged])) if flagged else None,
                    "flagged_mean_acc_after": float(np.mean([acc(r) for r, _ in flagged])) if flagged else None,
                }
            )
    res["harness_check"] = hc

    # 5. Did verify_first change how often agents query? (decoy code arms)
    beh = []
    for m in models:
        for p in PROTOCOLS:
            rs = [r for a in (CODE, REFUT) for r in idx.get((m, p, a), {}).values()]
            if rs:
                beh.append(
                    {
                        "model": m,
                        "protocol": p,
                        "n": len(rs),
                        "mean_queries": float(np.mean([r["queries_used"] for r in rs])),
                        "zero_query_share": float(np.mean([1.0 if r["queries_used"] == 0 else 0.0 for r in rs])),
                        "ge10_query_share": float(np.mean([1.0 if r["queries_used"] >= 10 else 0.0 for r in rs])),
                        "trap_rate": float(np.mean([trapped(r) for r in rs])),
                    }
                )
    res["behaviour_decoy_code"] = beh

    # 6. Replication: fresh standard protocol vs the locked records of the same model, arm and seed.
    lk: Dict[Tuple[str, str], Dict[int, Record]] = defaultdict(dict)
    for r in locked:
        if is_main(r):
            lk[(r["model"], arm(r))][int(r["seed"])] = r
    rep = []
    for a in ARMS_B:
        for metric_name, metric in (("acc", acc), ("trap", trapped)):
            if a in ("BB", CLEAN) and metric_name == "trap":
                continue
            per_model, diffs = [], []
            for m in models:
                A, B = idx.get((m, "standard", a), {}), lk.get((m, a), {})
                common = sorted(set(A) & set(B))
                if not common:
                    continue
                xa, xb = [metric(A[s]) for s in common], [metric(B[s]) for s in common]
                per_model.append({"model": m, **paired(xa, xb)})
                diffs.append(np.asarray(xa) - np.asarray(xb))
            if diffs:
                rep.append({"arm": a, "metric": metric_name, "per_model": per_model, "pooled": pooled_paired(diffs, seed=SEED + 13)})
    res["replication_fresh_minus_locked"] = rep

    # 7. Did trapped agents hold query results that contradict their own answer?
    ev: Dict[str, Any] = {}
    for p in PROTOCOLS:
        rs = [r for (mm, pp, aa), d in idx.items() if pp == p and aa in (CODE, REFUT) for r in d.values()]
        ev[p] = evidence_summary(rs)
        ev[p]["per_model"] = {
            m: evidence_summary([r for (mm, pp, aa), d in idx.items() if pp == p and mm == m and aa in (CODE, REFUT) for r in d.values()])["trapped"]
            for m in models
        }
    res["own_query_evidence_decoy_code"] = ev
    return res


# ==============================================================================
# PART C -- are decoy answers presented as guesses?
# ==============================================================================
HEDGE = re.compile(
    r"guess|could ?n[o']t|couldn.t|can.t (give|determine|confirm|verify|identify)|cannot (give|determine|confirm|verify|identify)"
    r"|unverified|not (a )?verified|low[- ]confidence|not confident|unable|fallback|fall back|probably (wrong|incorrect)"
    r"|did not (verify|confirm|determine|find)|not (been )?(able|confirmed)|uncertain|no tool calls left|ran out|out of (turns|tool)",
    re.I,
)
BARE_JSON_MAX_CHARS = 90  # a final message this short is the JSON answer with no explanation


def declared_guess_table(rs: Sequence[Record]) -> Dict[str, Any]:
    """Per model, on the two decoy-code variants: trapped answers and exact answers whose text matches HEDGE."""
    by: Dict[str, Dict[str, int]] = defaultdict(lambda: {"trapped": 0, "trapped_hedged": 0, "trapped_bare_json": 0, "exact": 0, "exact_hedged": 0})
    for r in rs:
        if r["mode"] != "WB" or r["variant"] not in (CODE, REFUT):
            continue
        txt = r.get("final_excerpt", "") or ""
        h = bool(HEDGE.search(txt))
        b = by[r["model"]]
        if trapped(r) >= 1.0:
            b["trapped"] += 1
            b["trapped_hedged"] += int(h)
            b["trapped_bare_json"] += int(len(txt.strip()) < BARE_JSON_MAX_CHARS)
        elif float(r.get("exact_rule", 0.0)) >= 1.0:
            b["exact"] += 1
            b["exact_hedged"] += int(h)
    total = {k: int(sum(b[k] for b in by.values())) for k in ("trapped", "trapped_hedged", "trapped_bare_json", "exact", "exact_hedged")}
    return {"per_model": {m: by[m] for m in sorted(by)}, "total": total}


def has_python_error(r: Record) -> bool:
    return any(t == "run_python:err" for t in r.get("tool_sequence", []))


def python_error_vs_trap(rs: Sequence[Record]) -> Dict[str, Any]:
    """Trap rate in decoy-code trials with at least one failed Python call versus trials with none.

    Stratified by cell (model, effort, budget profile, task, token ceiling); only cells holding
    both kinds of trial contribute to the stratified difference."""
    by: Dict[Any, Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
    n_err = n_ok = 0
    for r in rs:
        if r["mode"] != "WB" or r["variant"] not in (CODE, REFUT):
            continue
        if has_python_error(r):
            by[cell_key(r)][0].append(trapped(r))
            n_err += 1
        else:
            by[cell_key(r)][1].append(trapped(r))
            n_ok += 1
    strata = {k: (np.asarray(a), np.asarray(b)) for k, (a, b) in by.items() if a and b}
    out = strat_two_group(strata, seed=SEED + 21)
    out["n_trials_with_error"] = n_err
    out["n_trials_without_error"] = n_ok
    return out


def error_share_by_arm(rs: Sequence[Record]) -> Dict[str, Any]:
    by: Dict[str, List[float]] = defaultdict(list)
    for r in rs:
        by[arm(r)].append(1.0 if has_python_error(r) else 0.0)
    return {a: {"n": len(v), "share_with_python_error": float(np.mean(v))} for a, v in sorted(by.items())}


def part_c(locked: Sequence[Record], followup: Sequence[Record]) -> Dict[str, Any]:
    main_rows = [r for r in locked if is_main(r)]
    res = {
        "locked_main_condition": declared_guess_table(main_rows),
        "locked_all_exp2": declared_guess_table(locked),
        "python_error_share_main_condition": error_share_by_arm(main_rows),
        "python_error_vs_trap_main_condition": python_error_vs_trap(main_rows),
        "python_error_vs_trap_all_exp2": python_error_vs_trap(locked),
    }
    for p in PROTOCOLS:
        rs = [r for r in followup if proto(r) == p]
        if rs:
            res[f"followup_{p}"] = declared_guess_table(rs)
    return res


# ==============================================================================
# TEXT REPORT
# ==============================================================================
def f3(x: Optional[float]) -> str:
    return "  n/a " if x is None else f"{x:+.3f}"


def ci(t: Sequence[float]) -> str:
    return f"[{t[0]:+.3f}, {t[1]:+.3f}]"


def print_pooled(title: str, p: Dict[str, Any]) -> None:
    eq = p["equivalence"]
    print(
        f"{title}: diff={p['mean_diff']:+.4f} over {p['n_pairs']} pairs / {p['n_cells']} cells | "
        f"strat 90% {ci(p['strat_ci90'])} 95% {ci(p['strat_ci95'])} | cluster 90% {ci(p['cluster_ci90'])} "
        f"95% {ci(p['cluster_ci95'])} | p={p['signflip_p_two_sided']:.4f} | "
        f"equiv±0.05 strat={eq['0.05']['stratified']} clus={eq['0.05']['cluster']} | "
        f"equiv±0.10 strat={eq['0.10']['stratified']} clus={eq['0.10']['cluster']}"
    )


def report_c(c: Dict[str, Any]) -> None:
    print("=" * 100)
    print("PART C -- decoy answers that the agent itself presents as a guess (keyword pattern on the final message)")
    print("=" * 100)
    print("failed Python calls, main condition, share of trials by arm:", json.dumps(c["python_error_share_main_condition"]))
    for k in ("python_error_vs_trap_main_condition", "python_error_vs_trap_all_exp2"):
        print(f"{k}: {json.dumps(c[k])}")
    for name, t in c.items():
        if "per_model" not in t:
            continue
        print(f"{name}: total {json.dumps(t['total'])}")
        for m, b in t["per_model"].items():
            print(
                f"  {m:28s} trapped={b['trapped']:3d} hedged={b['trapped_hedged']:3d} bare_json={b['trapped_bare_json']:3d} | "
                f"exact={b['exact']:3d} hedged={b['exact_hedged']:3d}"
            )


def report(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> None:
    if a is not None:
        print("=" * 100)
        print(f"PART A -- refutable control on additional models ({a['n_new_trials']} trials, ${a['cost_usd']:.2f})")
        print("=" * 100)
        for r in a["models"]:
            d, h, bb = r["refutable"], r["hard_guard_locked"], r["bb_locked"]
            print(
                f"{r['model']:28s} refutable: n={d['n']} acc={d['acc']:.3f} valid={d['valid_json']:.2f} "
                f"trap={d['trap_n']}/{d['n']} q={d['mean_queries']:.1f} | hard-guard(locked): acc={h['acc']:.3f} "
                f"trap={h['trap_n']}/{h['n']} | BB(locked)={bb['acc']:.3f}"
            )
            x = r["delta_wb_refutable"]
            y = r["acc_code_minus_refut"]
            z = r["trap_code_minus_refut"]
            print(
                f"{'':28s} dWB(refutable)={x['diff']:+.3f} {ci(x['ci95'])} p={x['p_signflip']:.4f} | "
                f"code-refut acc={y['diff']:+.3f} {ci(y['ci95'])} p={y['p_signflip']:.4f} | "
                f"trap={z['diff']:+.3f} {ci(z['ci95'])}"
            )
        print()
        print_pooled("acc  code-refut, new cells only    ", a["acc_code_minus_refut_new_cells_only"]["pooled"])
        print_pooled("trap code-refut, new cells only    ", a["trap_code_minus_refut_new_cells_only"]["pooled"])
        print_pooled(f"acc  code-refut, extended ({a['acc_code_minus_refut_extended']['n_models']} models)", a["acc_code_minus_refut_extended"]["pooled"])
        print_pooled(f"trap code-refut, extended ({a['trap_code_minus_refut_extended']['n_models']} models)", a["trap_code_minus_refut_extended"]["pooled"])
        print_pooled("acc  code-refut, extended, valid JSON only", a["acc_code_minus_refut_extended_valid_json_only"]["pooled"])
        print_pooled(f"acc  code-refut, main condition ({a['acc_code_minus_refut_main_condition']['n_models']} models)", a["acc_code_minus_refut_main_condition"]["pooled"])
        print_pooled(f"trap code-refut, main condition ({a['trap_code_minus_refut_main_condition']['n_models']} models)", a["trap_code_minus_refut_main_condition"]["pooled"])
        print("own-query evidence (new refutable trials):", json.dumps(a["own_query_evidence"]))
        print()
    if b is not None:
        print("=" * 100)
        print(f"PART B -- protocols ({b['n_trials']} trials, ${b['cost_usd']:.2f}); cost by model: " + ", ".join(f"{m}=${c:.2f}" for m, c in b["cost_by_model"].items()))
        print("=" * 100)
        print(f"{'model':24s} {'protocol':14s} {'arm':30s} {'n':>3s} {'acc':>6s} {'valid':>5s} {'trap':>6s} {'q':>5s} {'0q':>5s} {'turns':>5s} {'trunc':>5s} {'dWB':>7s} {'95% CI':>18s} {'p':>7s}")
        for c in b["cells"]:
            d = c.get("delta_wb")
            tail = f"{d['diff']:+7.3f} {ci(d['ci95']):>18s} {d['p_signflip']:7.4f}" if d else ""
            print(
                f"{c['model']:24s} {c['protocol']:14s} {c['arm']:30s} {c['n']:3d} {c['acc']:6.3f} {c['valid_json']:5.2f} "
                f"{c['trap_n']:3d}/{c['n']:<2d} {c['mean_queries']:5.1f} {c['zero_query_share']:5.2f} {c['mean_turns']:5.2f} "
                f"{c['truncated_trials']:5d} {tail}"
            )
        print("\nProtocol minus standard (paired by model and seed):")
        for e in b["protocol_minus_standard"]:
            print_pooled(f"  {e['protocol']:14s} {e['arm']:30s} {e['metric']:4s}", e["pooled"])
            print("      per model: " + "; ".join(f"{x['model']} {x['diff']:+.3f} {ci(x['ci95'])} p={x['p_signflip']:.3f}" for x in e["per_model"]))
        print("\nPooled Delta_WB within protocol (WB minus BB of the same protocol):")
        for e in b["delta_wb_pooled"]:
            print_pooled(f"  {e['protocol']:14s} {e['arm']:30s}", e["pooled"])
        print("\nHarness check:")
        for h in b["harness_check"]:
            print(
                f"  {h['model']:24s} {h['arm']:30s} n={h['n']:3d} first_acc={h['first_answer_acc']:.3f} final_acc={h['final_acc']:.3f} "
                f"pass/mismatch/invalid={h['check_pass']}/{h['check_mismatch']}/{h['check_invalid']} | first trapped={h['first_trapped']} "
                f"(flagged {h['first_trapped_flagged']}, still trapped {h['first_trapped_final_trapped']}, exact after {h['first_trapped_final_exact']}) | "
                f"first exact={h['first_exact']} (flagged {h['first_exact_flagged']}) | flagged={h['flagged']} exact after={h['flagged_final_exact']} "
                f"acc {f3(h['flagged_mean_acc_before'])} -> {f3(h['flagged_mean_acc_after'])}"
            )
        print("\nBehaviour on the two decoy-code variants:")
        for x in b["behaviour_decoy_code"]:
            print(
                f"  {x['model']:24s} {x['protocol']:14s} n={x['n']:3d} mean_q={x['mean_queries']:5.1f} zero_q={x['zero_query_share']:.2f} "
                f">=10q={x['ge10_query_share']:.2f} trap={x['trap_rate']:.2f}"
            )
        print("\nOwn-query evidence on the two decoy-code variants (trapped answers vs exact answers):")
        for p_, e in b["own_query_evidence_decoy_code"].items():
            print(f"  {p_:14s} trapped: {json.dumps(e['trapped'])}")
            print(f"  {'':14s} exact  : {json.dumps(e['exact'])}")
            for m_, t in e["per_model"].items():
                print(f"  {'':14s}   {m_}: n={t['n']} queried={t['queried']} >=10q={t['queried_ge10']} contradicted={t['contradicted_by_own_queries']}")
        print("\nReplication: fresh standard protocol minus locked records (paired by seed):")
        for e in b["replication_fresh_minus_locked"]:
            print_pooled(f"  {e['arm']:30s} {e['metric']:4s}", e["pooled"])
            print("      per model: " + "; ".join(f"{x['model']} {x['mean_a']:.3f} vs {x['mean_b']:.3f}" for x in e["per_model"]))



# ==============================================================================
# PART D -- answer check with tool-enabled revision turns
# ==============================================================================
def revision_tool_calls(r: Record) -> List[str]:
    seq = list(r.get("tool_sequence", []))
    return seq[seq.index("harness_check:revise") + 1 :] if "harness_check:revise" in seq else []


def part_d(locked: Sequence[Record], proto_rows: Sequence[Record], new: Sequence[Record]) -> Dict[str, Any]:
    res: Dict[str, Any] = {"n_trials": len(new), "cost_usd": float(sum(r["cost_usd"] for r in new))}
    idx: Dict[Tuple[str, str, str], Dict[int, Record]] = defaultdict(dict)
    for r in list(proto_rows) + list(new):
        assert is_main(r), r["run_key"]
        k = (r["model"], proto(r), arm(r))
        assert int(r["seed"]) not in idx[k], f"duplicate {r['run_key']}"
        idx[k][int(r["seed"])] = r
    assert all(proto(r) == "harness_check_tools" for r in new)
    all_models = sorted({r["model"] for r in new})
    arms = [a for a in ("BB", CODE) if any((m, "harness_check_tools", a) in idx for m in all_models)]
    # A model enters the analysis only if every arm has the full set of seeds. A cell cut short by
    # the provider (credits ran out) is listed under incomplete_cells and left out of every estimate.
    n_full = max(len(idx[(m, "harness_check_tools", a)]) for m in all_models for a in arms)
    models = [m for m in all_models if all(len(idx.get((m, "harness_check_tools", a), {})) == n_full for a in arms)]
    res["models"] = models
    res["seeds_per_cell"] = n_full
    res["cost_by_model"] = {m: float(sum(r["cost_usd"] for r in new if r["model"] == m)) for m in all_models}
    res["n_trials_analysed"] = int(sum(len(idx[(m, "harness_check_tools", a)]) for m in models for a in arms))
    res["incomplete_cells"] = [
        {"model": m, "arm": a, **describe(list(idx[(m, "harness_check_tools", a)].values()))}
        for m in all_models
        if m not in models
        for a in arms
        if idx.get((m, "harness_check_tools", a))
    ]

    cells = []
    for m in models + ["ALL"]:
        for a in arms:
            rs = [
                r
                for (mm, p, aa), d in idx.items()
                if p == "harness_check_tools" and aa == a and mm in models and (m == "ALL" or mm == m)
                for r in d.values()
            ]
            if not rs:
                continue
            first = [first_answer_score(r) for r in rs]
            flagged = [(r, f) for r, f in zip(rs, first) if r.get("check_result") in ("mismatch", "invalid")]
            first_trap = [(r, f) for r, f in zip(rs, first) if f["matches_decoy"] >= 1.0]
            first_exact = [(r, f) for r, f in zip(rs, first) if f["exact_rule"] >= 1.0]
            rev_q = [r["queries_used"] - r.get("queries_before_check", 0) - r.get("check_queries", 0) for r, _ in flagged]
            cells.append(
                {
                    "model": m,
                    "arm": a,
                    **describe(rs),
                    "first_answer_acc": float(np.mean([f["heldout_acc"] for f in first])),
                    "first_trapped": len(first_trap),
                    "first_trapped_flagged": int(sum(1 for r, _ in first_trap if r.get("check_result") == "mismatch")),
                    "first_trapped_final_trapped": int(sum(1 for r, _ in first_trap if trapped(r))),
                    "first_trapped_final_exact": int(sum(1 for r, _ in first_trap if r.get("exact_rule", 0.0) >= 1.0)),
                    "first_exact": len(first_exact),
                    "first_exact_flagged": int(sum(1 for r, _ in first_exact if r.get("check_result") != "pass")),
                    "check_pass": int(sum(1 for r in rs if r.get("check_result") == "pass")),
                    "check_mismatch": int(sum(1 for r in rs if r.get("check_result") == "mismatch")),
                    "check_invalid": int(sum(1 for r in rs if r.get("check_result") == "invalid")),
                    "flagged": len(flagged),
                    "flagged_final_exact": int(sum(1 for r, _ in flagged if r.get("exact_rule", 0.0) >= 1.0)),
                    "flagged_final_valid": int(sum(1 for r, _ in flagged if r.get("parse_ok"))),
                    "flagged_mean_acc_before": float(np.mean([f["heldout_acc"] for _, f in flagged])) if flagged else None,
                    "flagged_mean_acc_after": float(np.mean([acc(r) for r, _ in flagged])) if flagged else None,
                    "flagged_used_tools": int(sum(1 for r, _ in flagged if revision_tool_calls(r))),
                    "flagged_ran_python": int(
                        sum(1 for r, _ in flagged if any(t.startswith("run_python") for t in revision_tool_calls(r)))
                    ),
                    "flagged_mean_new_queries": float(np.mean(rev_q)) if rev_q else None,
                    "flagged_truncated": int(sum(1 for r, _ in flagged if r.get("truncated_turns", 0) > 0)),
                }
            )
    res["cells"] = cells

    # Paired by seed against the two earlier protocols of the same model and arm.
    effects = []
    for ref in ("standard", "harness_check"):
        for a in arms:
            for metric_name, metric in (("acc", acc), ("trap", trapped), ("turns", lambda r: float(r["turns"]))):
                if a == "BB" and metric_name == "trap":
                    continue
                per_model, diffs = [], []
                for m in models:
                    A, B = idx.get((m, "harness_check_tools", a), {}), idx.get((m, ref, a), {})
                    seeds = sorted(set(A) & set(B))
                    if not seeds:
                        continue
                    xa, xb = [metric(A[s]) for s in seeds], [metric(B[s]) for s in seeds]
                    per_model.append({"model": m, **paired(xa, xb)})
                    diffs.append(np.asarray(xa) - np.asarray(xb))
                if diffs:
                    effects.append(
                        {
                            "reference": ref,
                            "arm": a,
                            "metric": metric_name,
                            "per_model": per_model,
                            "pooled": pooled_paired(diffs, seed=SEED + 21),
                        }
                    )
    res["minus_reference"] = effects

    # Locked 8-turn records (two Claude models only): BB from the 8/8-turn condition, white-box
    # code from the condition that gives the white-box mode 8 turns.
    track_8 = {"BB": "matched_wide_equal_budget", CODE: "matched_query_free_code"}
    wide = []
    for m in models:
        for a in arms:
            rs = [
                r
                for r in locked
                if r["model"] == m
                and r["budget_track"] == track_8[a]
                and r.get("task_type", "majority3") == "majority3"
                and r.get("effort") == "medium"
                and mt_override(r) == 0
                and arm(r) == a
            ]
            if rs:
                wide.append({"model": m, "arm": a, **describe(rs)})
    res["locked_8_turns"] = wide
    return res


def report_d(d: Dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print(
        f"PART D -- answer check with tool-enabled revision turns ({d['n_trials']} trials, {d['n_trials_analysed']} analysed, ${d['cost_usd']:.2f}); cost by model: "
        + ", ".join(f"{m}=${c:.2f}" for m, c in d["cost_by_model"].items())
    )
    print("=" * 100)
    print(f"{'model':24s} {'arm':12s} {'n':>3s} {'first':>6s} {'final':>6s} {'trap1':>5s} {'trapF':>5s} {'flag':>4s} {'fixed':>5s} {'tools':>5s} {'newq':>5s} {'turns':>5s} {'valid':>5s}")
    for c in d["cells"]:
        nq = c["flagged_mean_new_queries"]
        print(
            f"{c['model']:24s} {c['arm']:12s} {c['n']:3d} {c['first_answer_acc']:6.3f} {c['acc']:6.3f} {c['first_trapped']:5d} {c['trap_n']:5d} "
            f"{c['flagged']:4d} {c['flagged_final_exact']:5d} {c['flagged_used_tools']:5d} {(nq if nq is not None else 0.0):5.1f} {c['mean_turns']:5.2f} {c['valid_json']:5.2f}"
        )
    for c in d["incomplete_cells"]:
        print(f"  incomplete, not analysed: {c['model']:22s} {c['arm']:12s} n={c['n']:2d} of {d['seeds_per_cell']} acc={c['acc']:.3f} trap={c['trap_n']}")
    print("\nCheck with tools minus reference protocol (paired by seed; pooled over models):")
    for e in d["minus_reference"]:
        print_pooled(f"  vs {e['reference']:14s} {e['arm']:12s} {e['metric']:5s}", e["pooled"])
        print("      per model: " + "; ".join(f"{x['model']} {x['mean_a']:.3f} vs {x['mean_b']:.3f}" for x in e["per_model"]))
    print("\nLocked 8-turn condition (same models where it was run):")
    for c in d["locked_8_turns"]:
        print(
            f"  {c['model']:24s} {c['arm']:12s} n={c['n']:3d} acc={c['acc']:.3f} trap={c['trap_n']:2d} turns={c['mean_turns']:.2f} "
            f"cost/trial=${c['cost_usd'] / c['n']:.4f}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-json", default="", help="Write the full result structure to this JSON file.")
    ap.add_argument("--refutable", default=str(REFUTABLE_FILE))
    ap.add_argument("--protocols", default=str(PROTOCOLS_FILE))
    ap.add_argument("--check-tools", default=str(CHECK_TOOLS_FILE))
    args = ap.parse_args(argv)

    locked = load_exp2()
    out: Dict[str, Any] = {"n_locked_exp2_records": len(locked)}
    pa = pb = None
    if Path(args.refutable).exists():
        pa = part_a(locked, load_jsonl(Path(args.refutable)))
        out["part_a_refutable_extension"] = pa
    if Path(args.protocols).exists():
        pb = part_b(locked, load_jsonl(Path(args.protocols)))
        out["part_b_protocols"] = pb
    followup_rows: List[Record] = []
    for f in (args.refutable, args.protocols):
        if Path(f).exists():
            followup_rows.extend(load_jsonl(Path(f)))
    pc = part_c(locked, followup_rows)
    out["part_c_declared_guesses"] = pc
    pd_ = None
    if Path(args.check_tools).exists() and Path(args.protocols).exists():
        pd_ = part_d(locked, load_jsonl(Path(args.protocols)), load_jsonl(Path(args.check_tools)))
        out["part_d_check_with_tools"] = pd_
    report(pa, pb)
    report_c(pc)
    if pd_ is not None:
        report_d(pd_)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
