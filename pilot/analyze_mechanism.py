"""Mechanism analysis for Experiment 2 (computational Blackwell inversion).

Part 1 -- Equivalence of anti-transparent code (`wb_inv_code`) and the
byte-identical refutable control (`wb_decoy_long_refutable_guard`):
paired per-cell differences, a pooled paired difference with a bootstrap
stratified by cell (cells fixed) and a two-stage cluster bootstrap (cells
resampled), and a TOST-style verdict (equivalent at margin d iff the 90% CI
lies inside (-d, +d)). Same analysis for the trap rate, a sensitivity
contrast (refutable control minus short control), and a valid-JSON-only
variant of the main contrast.

Part 2 -- What the agents did, from per-trial telemetry (queries_used,
python_calls, turns, tool_sequence, thinking_tokens, truncated_turns):
per-model behaviour in the main condition, how opaque-weight agents reach an
answer, and trapped-vs-untrapped contrasts on anti-transparent code that are
stratified by model (main condition) or by cell (all Exp 2 cells).

Part 3 -- Offline replay of a verification-first arbitration rule: regenerate
each instance, check the WB answer against the oracle function on k fresh
random inputs, keep it if all agree and otherwise fall back to the BB answer
of the same seed and cell.

Scoring convention (inherited from the scorer in baselines.py and used by
analyze.py / analyze_c3_unified.py): a trial whose final answer is not a valid
rule scores heldout_acc = 0.50 and matches_decoy = 0.

Run from the repository root:
    python3 -m pilot.analyze_mechanism --out-json pilot/report_mechanism.json \
        > pilot/report_mechanism.txt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .analyze import load_jsonl, paired_delta_stats

ROOT = Path(__file__).resolve().parent
SOURCES = [ROOT / "results_c2_pilot.jsonl", ROOT / "results_c3_expansion.jsonl"]

N_BOOT = 10_000
N_PERM = 10_000
SEED = 2026

CODE = "wb_inv_code"
REFUT = "wb_decoy_long_refutable_guard"
SHORT = "wb_decoy_trivial_guard"
CLEAN = "wb_clean"
RELU = "wb_inv_relu"
MARGINS = (0.05, 0.10)

# Expected structure (sanity assertions).
EXPECTED_EXP2_RECORDS = 4379
EXPECTED_PAIR_CELLS = 18
EXPECTED_PAIRS = 405
EXPECTED_MAIN_PER_ARM = 345
EXPECTED_MAIN_MODELS = 15
EXPECTED_MAIN_REFUT = 220  # refutable control was run on only a subset of models

Cell = Tuple[str, str, str, str, int]
Record = Dict[str, Any]


# ==============================================================================
# LOADING / KEYS
# ==============================================================================
def load_exp2() -> List[Record]:
    rows: List[Record] = []
    for src in SOURCES:
        rows.extend(r for r in load_jsonl(src) if r.get("experiment") == "exp2_blackwell")
    return rows


def mt_override(r: Record) -> int:
    v = r.get("max_tokens_override")
    if v:
        return int(v)
    return 8192 if "|mt=8192" in r.get("run_key", "") else 0


def cell_key(r: Record) -> Cell:
    """(model, effort, budget_track, task_type, max_tokens_override)."""
    return (r["model"], r["effort"], r["budget_track"], r["task_type"], mt_override(r))


def arm(r: Record) -> str:
    """BB records carry variant 'wb_inv_code' in the logs; label them 'BB'."""
    return "BB" if r["mode"] == "BB" else r["variant"]


def is_main(r: Record) -> bool:
    return (
        r["budget_track"] == "matched_medium_budget"
        and r["task_type"] == "majority3"
        and r["effort"] in ("medium", "default")
        and mt_override(r) == 0
    )


def oracle_calls(r: Record) -> int:
    return sum(1 for t in r.get("tool_sequence", []) if t.startswith("query_oracle"))


def python_calls_seq(r: Record) -> int:
    return sum(1 for t in r.get("tool_sequence", []) if t.startswith("run_python"))


def oracle_after_first_python(r: Record) -> Optional[int]:
    """1 if any oracle call is listed after the first run_python call, 0 if not,
    None if the run made no run_python call (ordering undefined)."""
    seq = [t.split(":")[0] for t in r.get("tool_sequence", [])]
    if "run_python" not in seq:
        return None
    i = seq.index("run_python")
    return int(any(t.startswith("query_oracle") for t in seq[i + 1 :]))


def oracle_before_first_python(r: Record) -> Optional[int]:
    seq = [t.split(":")[0] for t in r.get("tool_sequence", [])]
    if "run_python" not in seq:
        return None
    i = seq.index("run_python")
    return int(any(t.startswith("query_oracle") for t in seq[:i]))


def single_tool_batch(r: Record) -> bool:
    """True if all tool calls must have been issued in one assistant turn.

    The last turn is always tool-free (either the model answered or tool_choice
    was 'none'), so tool-using turns <= turns - 1. With turns <= 2 every tool
    call shares one batch and its listed order says nothing about whether the
    agent saw a Python result before querying."""
    return int(r.get("turns", 0)) <= 2


# ==============================================================================
# STATISTICS
# ==============================================================================
def _q(a: np.ndarray, lo: float, hi: float) -> Tuple[float, float]:
    return float(np.quantile(a, lo)), float(np.quantile(a, hi))


def pooled_paired(cell_diffs: Sequence[np.ndarray], seed: int = SEED, n_boot: int = N_BOOT) -> Dict[str, Any]:
    """Pooled (pair-weighted) mean paired difference across cells.

    - stratified bootstrap: resample pairs within each cell, cells fixed.
    - cluster bootstrap: resample cells with replacement, then pairs within each
      drawn cell (two-stage); ratio estimator sum(diffs) / sum(n).
    - sign-flip permutation p-value (two-sided) on all pairs.
    """
    diffs = [np.asarray(d, dtype=np.float64) for d in cell_diffs if len(d) > 0]
    k = len(diffs)
    ns = np.array([len(d) for d in diffs], dtype=np.float64)
    n_tot = float(ns.sum())
    allv = np.concatenate(diffs)
    obs = float(allv.mean())
    rng = np.random.default_rng(seed)

    strat_sums = np.zeros(n_boot)
    for d in diffs:
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        strat_sums += d[idx].sum(axis=1)
    strat = strat_sums / n_tot

    # Two-stage cluster bootstrap.
    pick = rng.integers(0, k, size=(n_boot, k))
    clus_sums = np.zeros(n_boot)
    for c, d in enumerate(diffs):
        mask = pick == c  # (B, K) positions that drew cell c
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        idx = rng.integers(0, len(d), size=(cnt, len(d)))
        draws = d[idx].sum(axis=1)
        rows = np.nonzero(mask)[0]
        np.add.at(clus_sums, rows, draws)
    clus_n = ns[pick].sum(axis=1)
    clus = clus_sums / clus_n

    signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(allv)))
    perm = (signs * allv[None, :]).mean(axis=1)
    p_two = float((1 + np.sum(np.abs(perm) >= abs(obs) - 1e-12)) / (N_PERM + 1))

    cell_means = np.array([d.mean() for d in diffs])
    out = {
        "n_cells": k,
        "n_pairs": int(n_tot),
        "mean_diff": obs,
        "mean_of_cell_means": float(cell_means.mean()),
        "strat_ci90": _q(strat, 0.05, 0.95),
        "strat_ci95": _q(strat, 0.025, 0.975),
        "cluster_ci90": _q(clus, 0.05, 0.95),
        "cluster_ci95": _q(clus, 0.025, 0.975),
        "signflip_p_two_sided": p_two,
        "equivalence": {},
    }
    for d in MARGINS:
        out["equivalence"][f"{d:.2f}"] = {
            "stratified": bool(-d < out["strat_ci90"][0] and out["strat_ci90"][1] < d),
            "cluster": bool(-d < out["cluster_ci90"][0] and out["cluster_ci90"][1] < d),
        }
    return out


def strat_two_group(
    strata: Dict[Any, Tuple[np.ndarray, np.ndarray]], seed: int = SEED
) -> Dict[str, Any]:
    """Stratified difference mean(group A) - mean(group B).

    Pooled estimate is the CMH-style weighted average of within-stratum
    differences with weights n_a * n_b / (n_a + n_b). Inference: permutation of
    group labels within strata (two-sided) and a bootstrap resampling each
    group within each stratum. Also returns the naive unstratified difference.
    """
    keys = sorted(strata.keys(), key=str)
    rng = np.random.default_rng(seed)
    w, dif = [], []
    perm_num = np.zeros(N_PERM)
    boot_num = np.zeros(N_BOOT)
    all_a, all_b = [], []
    for s in keys:
        a, b = (np.asarray(x, dtype=np.float64) for x in strata[s])
        na, nb = len(a), len(b)
        ws = na * nb / (na + nb)
        w.append(ws)
        dif.append(a.mean() - b.mean())
        all_a.append(a)
        all_b.append(b)
        v = np.concatenate([a, b])
        order = np.argsort(rng.random((N_PERM, na + nb)), axis=1)
        pv = v[order]
        perm_num += ws * (pv[:, :na].mean(axis=1) - pv[:, na:].mean(axis=1))
        ia = rng.integers(0, na, size=(N_BOOT, na))
        ib = rng.integers(0, nb, size=(N_BOOT, nb))
        boot_num += ws * (a[ia].mean(axis=1) - b[ib].mean(axis=1))
    wsum = float(np.sum(w))
    obs = float(np.dot(w, dif) / wsum)
    perm = perm_num / wsum
    boot = boot_num / wsum
    p = float((1 + np.sum(np.abs(perm) >= abs(obs) - 1e-12)) / (N_PERM + 1))
    ca, cb = np.concatenate(all_a), np.concatenate(all_b)
    return {
        "n_strata": len(keys),
        "n_a": int(len(ca)),
        "n_b": int(len(cb)),
        "mean_a": float(ca.mean()),
        "mean_b": float(cb.mean()),
        "naive_diff": float(ca.mean() - cb.mean()),
        "strat_diff": obs,
        "ci95": _q(boot, 0.025, 0.975),
        "perm_p_two_sided": p,
    }


# ==============================================================================
# PART 1
# ==============================================================================
def build_pairs(rows: Sequence[Record]) -> Dict[Cell, Dict[str, Dict[int, Record]]]:
    by: Dict[Cell, Dict[str, Dict[int, Record]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["mode"] != "WB":
            continue
        seeds = by[cell_key(r)][r["variant"]]
        assert int(r["seed"]) not in seeds, f"duplicate trial {r['run_key']}"
        seeds[int(r["seed"])] = r
    return by


def contrast(
    by: Dict[Cell, Dict[str, Dict[int, Record]]],
    va: str,
    vb: str,
    metric: Callable[[Record], float],
    pair_filter: Optional[Callable[[Record, Record], bool]] = None,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    cells_out, diffs = [], []
    for c in sorted(by):
        ma, mb = by[c].get(va, {}), by[c].get(vb, {})
        common = sorted(set(ma) & set(mb))
        if pair_filter is not None:
            common = [s for s in common if pair_filter(ma[s], mb[s])]
        if not common:
            continue
        xa = [metric(ma[s]) for s in common]
        xb = [metric(mb[s]) for s in common]
        st = paired_delta_stats(xa, xb, alternative="two-sided", n_perm=N_PERM, n_boot=N_BOOT, seed=SEED)
        cells_out.append(
            {
                "cell": list(c),
                "n_pairs": st["n_pairs"],
                "mean_a": st["mean_wb"],
                "mean_b": st["mean_bb"],
                "diff": st["delta_wb"],
                "ci95": [st["ci_lo"], st["ci_hi"]],
                "p_signflip": st["p_value"],
            }
        )
        diffs.append(np.asarray(xa) - np.asarray(xb))
    return cells_out, diffs


def acc(r: Record) -> float:
    return float(r["heldout_acc"])


def trapped(r: Record) -> float:
    return 1.0 if r.get("matches_decoy") else 0.0


def both_valid(a: Record, b: Record) -> bool:
    return bool(a.get("parse_ok")) and bool(b.get("parse_ok"))


def part1(rows: Sequence[Record]) -> Dict[str, Any]:
    by = build_pairs(rows)
    res: Dict[str, Any] = {}

    cells, diffs = contrast(by, CODE, REFUT, acc)
    assert len(cells) == EXPECTED_PAIR_CELLS, f"expected {EXPECTED_PAIR_CELLS} cells, got {len(cells)}"
    assert sum(c["n_pairs"] for c in cells) == EXPECTED_PAIRS, "unexpected number of code/refutable pairs"
    res["acc_code_minus_refut"] = {"cells": cells, "pooled": pooled_paired(diffs, seed=SEED)}

    cells_t, diffs_t = contrast(by, CODE, REFUT, trapped)
    res["trap_code_minus_refut"] = {"cells": cells_t, "pooled": pooled_paired(diffs_t, seed=SEED + 1)}

    # Sensitivity contrast restricted to the same 18 cells (cells holding the refutable control).
    refut_cells = {tuple(c["cell"]) for c in cells}
    by_r = {c: v for c, v in by.items() if c in refut_cells}
    cells_s, diffs_s = contrast(by_r, REFUT, SHORT, acc)
    res["acc_refut_minus_short"] = {"cells": cells_s, "pooled": pooled_paired(diffs_s, seed=SEED + 2)}
    cells_st, diffs_st = contrast(by_r, REFUT, SHORT, trapped)
    res["trap_refut_minus_short"] = {"cells": cells_st, "pooled": pooled_paired(diffs_st, seed=SEED + 3)}

    cells_v, diffs_v = contrast(by, CODE, REFUT, acc, pair_filter=both_valid)
    res["acc_code_minus_refut_valid_json_only"] = {"cells": cells_v, "pooled": pooled_paired(diffs_v, seed=SEED + 4)}

    # Parse-failure counts behind the 0.50 convention.
    pf = {"code_invalid": 0, "refut_invalid": 0, "pairs_any_invalid": 0}
    for c in refut_cells:
        ma, mb = by[c][CODE], by[c][REFUT]
        for s in set(ma) & set(mb):
            ia, ib = not ma[s].get("parse_ok"), not mb[s].get("parse_ok")
            pf["code_invalid"] += int(ia)
            pf["refut_invalid"] += int(ib)
            pf["pairs_any_invalid"] += int(ia or ib)
    res["parse_failures_in_pairs"] = pf
    return res


# ==============================================================================
# PART 2
# ==============================================================================
ARMS = ["BB", CLEAN, SHORT, REFUT, CODE, RELU]
ARM_SHORT = {"BB": "BB", CLEAN: "clean", SHORT: "short", REFUT: "refut", CODE: "code", RELU: "weights"}


def describe(rs: Sequence[Record]) -> Dict[str, Any]:
    q = np.array([r["queries_used"] for r in rs], dtype=np.float64)
    return {
        "n": len(rs),
        "mean_queries": float(q.mean()),
        "median_queries": float(np.median(q)),
        "zero_query_share": float(np.mean(q == 0)),
        "mean_python_calls": float(np.mean([r["python_calls"] for r in rs])),
        "mean_turns": float(np.mean([r["turns"] for r in rs])),
        "trap_rate": float(np.mean([trapped(r) for r in rs])),
        "mean_acc": float(np.mean([acc(r) for r in rs])),
        "valid_json_rate": float(np.mean([1.0 if r.get("parse_ok") else 0.0 for r in rs])),
    }


def relu_routes(rs: Sequence[Record]) -> Dict[str, Any]:
    """Classify opaque-weight runs by the tools they attempted."""
    groups: Dict[str, List[Record]] = defaultdict(list)
    for r in rs:
        no, npy = oracle_calls(r), python_calls_seq(r)
        if no == 0 and npy == 0:
            g = "no_tools"
        elif no == 0:
            g = "python_only"
        elif npy == 0:
            g = "oracle_only"
        else:
            g = "python_and_oracle"
        groups[g].append(r)
    n = len(rs)
    out: Dict[str, Any] = {"n": n, "groups": {}}
    for g in ("python_only", "python_and_oracle", "oracle_only", "no_tools"):
        gr = groups.get(g, [])
        out["groups"][g] = {
            "n": len(gr),
            "share": len(gr) / n if n else 0.0,
            "mean_acc": float(np.mean([acc(r) for r in gr])) if gr else None,
            "exact_rule_rate": float(np.mean([r.get("exact_rule", 0.0) for r in gr])) if gr else None,
            "valid_json_rate": float(np.mean([1.0 if r.get("parse_ok") else 0.0 for r in gr])) if gr else None,
        }
    zq = [r for r in rs if r["queries_used"] == 0]
    pq = [r for r in rs if r["queries_used"] > 0]
    out["zero_oracle_queries"] = {
        "n": len(zq),
        "share": len(zq) / n if n else 0.0,
        "mean_acc": float(np.mean([acc(r) for r in zq])) if zq else None,
    }
    out["some_oracle_queries"] = {
        "n": len(pq),
        "share": len(pq) / n if n else 0.0,
        "mean_acc": float(np.mean([acc(r) for r in pq])) if pq else None,
    }
    return out


TT_METRICS: Dict[str, Callable[[Record], Optional[float]]] = {
    "queries_used": lambda r: float(r["queries_used"]),
    "zero_query": lambda r: 1.0 if r["queries_used"] == 0 else 0.0,
    "python_calls": lambda r: float(r["python_calls"]),
    "turns": lambda r: float(r["turns"]),
    "oracle_after_first_python": lambda r: (None if oracle_after_first_python(r) is None else float(oracle_after_first_python(r))),
    "oracle_before_first_python": lambda r: (None if oracle_before_first_python(r) is None else float(oracle_before_first_python(r))),
    "thinking_tokens": lambda r: float(r.get("thinking_tokens", 0)),
    "any_truncated_turn": lambda r: 1.0 if r.get("truncated_turns", 0) > 0 else 0.0,
    "truncated_turns": lambda r: float(r.get("truncated_turns", 0)),
}


def trapped_vs_untrapped(
    rs: Sequence[Record], stratum: Callable[[Record], Any], min_each: int = 3, seed: int = SEED
) -> Dict[str, Any]:
    """Trapped (A) minus untrapped (B) on anti-transparent-code runs, stratified."""
    by: Dict[Any, List[Record]] = defaultdict(list)
    for r in rs:
        by[stratum(r)].append(r)
    out: Dict[str, Any] = {"min_each": min_each, "metrics": {}, "per_stratum": []}
    elig_all = {}
    for s, g in by.items():
        t = [r for r in g if trapped(r)]
        u = [r for r in g if not trapped(r)]
        if len(t) >= min_each and len(u) >= min_each:
            elig_all[s] = (t, u)
    out["n_strata_total"] = len(by)
    out["n_strata_eligible"] = len(elig_all)
    out["n_trapped_eligible"] = sum(len(t) for t, _ in elig_all.values())
    out["n_untrapped_eligible"] = sum(len(u) for _, u in elig_all.values())
    out["n_trapped_all"] = sum(1 for r in rs if trapped(r))
    out["n_untrapped_all"] = sum(1 for r in rs if not trapped(r))
    for s in sorted(elig_all, key=str):
        t, u = elig_all[s]
        out["per_stratum"].append(
            {
                "stratum": list(s) if isinstance(s, tuple) else s,
                "n_trapped": len(t),
                "n_untrapped": len(u),
                "mean_q_trapped": float(np.mean([r["queries_used"] for r in t])),
                "mean_q_untrapped": float(np.mean([r["queries_used"] for r in u])),
                "zero_q_trapped": float(np.mean([r["queries_used"] == 0 for r in t])),
                "zero_q_untrapped": float(np.mean([r["queries_used"] == 0 for r in u])),
            }
        )
    for i, (name, fn) in enumerate(TT_METRICS.items()):
        strata: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}
        for s, (t, u) in elig_all.items():
            if name == "thinking_tokens":
                # Some providers do not report thinking tokens; skip strata with none reported.
                if not any(r.get("thinking_tokens", 0) > 0 for r in t + u):
                    continue
            a = [fn(r) for r in t]
            b = [fn(r) for r in u]
            a = [x for x in a if x is not None]
            b = [x for x in b if x is not None]
            if len(a) >= 1 and len(b) >= 1:
                strata[s] = (np.array(a), np.array(b))
        if strata:
            out["metrics"][name] = strat_two_group(strata, seed=seed + 17 * (i + 1))
    return out


def part2(rows: Sequence[Record]) -> Dict[str, Any]:
    main = [r for r in rows if is_main(r)]
    per_arm: Dict[str, List[Record]] = defaultdict(list)
    for r in main:
        per_arm[arm(r)].append(r)
    counts = {a: len(per_arm[a]) for a in ARMS}
    models = sorted({r["model"] for r in main})
    for a in (CLEAN, SHORT, CODE, RELU, "BB"):
        assert counts[a] == EXPECTED_MAIN_PER_ARM, f"main condition {a}: {counts[a]} != {EXPECTED_MAIN_PER_ARM}"
    assert counts[REFUT] == EXPECTED_MAIN_REFUT, f"main condition refutable: {counts[REFUT]}"
    assert len(models) == EXPECTED_MAIN_MODELS, f"main condition models: {len(models)}"
    refut_models = sorted({r["model"] for r in per_arm[REFUT]})

    table: Dict[str, Dict[str, Any]] = {}
    for m in models + ["ALL"]:
        table[m] = {}
        for a in ARMS:
            rs = [r for r in per_arm[a] if m == "ALL" or r["model"] == m]
            if rs:
                table[m][a] = describe(rs)

    relu_main = relu_routes(per_arm[RELU])
    relu_by_model = {m: relu_routes([r for r in per_arm[RELU] if r["model"] == m]) for m in models}
    code_main_routes = relu_routes(per_arm[CODE])  # same classifier, for comparison
    bb_main_routes = relu_routes(per_arm["BB"])

    code_main = per_arm[CODE]
    tt_main = trapped_vs_untrapped(code_main, lambda r: r["model"], seed=SEED + 100)

    code_all = [r for r in rows if r["mode"] == "WB" and r["variant"] == CODE]
    tt_all = trapped_vs_untrapped(code_all, cell_key, seed=SEED + 200)

    # Ordering-granularity diagnostics for anti-transparent-code runs.
    def order_diag(rs: Sequence[Record]) -> Dict[str, Any]:
        withpy = [r for r in rs if python_calls_seq(r) > 0]
        after = [r for r in withpy if oracle_after_first_python(r)]
        return {
            "n_runs": len(rs),
            "n_with_python": len(withpy),
            "n_oracle_after_first_python": len(after),
            "share_single_tool_batch_all": float(np.mean([single_tool_batch(r) for r in rs])) if rs else None,
            "share_single_tool_batch_among_after": float(np.mean([single_tool_batch(r) for r in after])) if after else None,
            "trapped": {
                "n_with_python": sum(1 for r in withpy if trapped(r)),
                "share_oracle_after_first_python": float(np.mean([bool(oracle_after_first_python(r)) for r in withpy if trapped(r)]))
                if any(trapped(r) for r in withpy)
                else None,
            },
            "untrapped": {
                "n_with_python": sum(1 for r in withpy if not trapped(r)),
                "share_oracle_after_first_python": float(np.mean([bool(oracle_after_first_python(r)) for r in withpy if not trapped(r)]))
                if any(not trapped(r) for r in withpy)
                else None,
            },
        }

    # Trapped runs that did query the oracle (verification did not prevent the trap).
    def trapped_with_queries(rs: Sequence[Record]) -> Dict[str, Any]:
        t = [r for r in rs if trapped(r)]
        return {
            "n_trapped": len(t),
            "n_trapped_with_queries": sum(1 for r in t if r["queries_used"] > 0),
            "n_trapped_with_ge10_queries": sum(1 for r in t if r["queries_used"] >= 10),
        }

    return {
        "main_counts": counts,
        "main_models": models,
        "refut_models_main": refut_models,
        "table": table,
        "relu_routes_main": relu_main,
        "relu_routes_by_model": relu_by_model,
        "code_routes_main": code_main_routes,
        "bb_routes_main": bb_main_routes,
        "trapped_vs_untrapped_main": tt_main,
        "trapped_vs_untrapped_all_cells": tt_all,
        "ordering_main": order_diag(code_main),
        "ordering_all_cells": order_diag(code_all),
        "trapped_with_queries_main": trapped_with_queries(code_main),
        "trapped_with_queries_all_cells": trapped_with_queries(code_all),
    }


# ==============================================================================
# PART 3: OFFLINE REPLAY OF A VERIFICATION-FIRST ARBITRATION RULE
# ==============================================================================
ARB_KS = (5, 10, 20)
ARB_SEED_TAG = "arbitration_check"
WB_VARIANTS = [CODE, REFUT, SHORT, RELU, CLEAN]

_INST_CACHE: Dict[Tuple[int, str, str], Any] = {}
_CHECK_CACHE: Dict[Tuple[int, str, str], Tuple[np.ndarray, np.ndarray]] = {}


def instance_for(r: Record) -> Any:
    """Regenerate the instance exactly as runner.run_exp2_trial does."""
    from .generators import generate_blackwell_inv_instance

    key = (int(r["seed"]), r["task_type"], r["variant"])
    if key not in _INST_CACHE:
        _INST_CACHE[key] = generate_blackwell_inv_instance(
            n=int(r.get("n", 40)), support_size=3, variant=r["variant"], seed=int(r["seed"]), task_type=r["task_type"]
        )
    return _INST_CACHE[key]


def check_points(r: Record) -> Tuple[np.ndarray, np.ndarray]:
    """max(ARB_KS) fresh uniform inputs for this instance and the oracle's answers on them.

    The RNG stream is derived from (seed, task_type) with a tag distinct from the
    held-out stream, so the points are independent of the scoring points; the
    first k rows are used for budget k (nested checks). Labels come from
    eval_inv, i.e. exactly what query_oracle would have returned."""
    from .generators import _deterministic_seed

    key = (int(r["seed"]), r["task_type"], r["variant"])
    if key not in _CHECK_CACHE:
        inst = instance_for(r)
        rng = np.random.default_rng(_deterministic_seed(int(r["seed"]), f"{ARB_SEED_TAG}_{r['task_type']}"))
        X = rng.choice([-1, 1], size=(max(ARB_KS), inst.n)).astype(np.int8)
        _CHECK_CACHE[key] = (X, inst.eval_inv(X))
    return _CHECK_CACHE[key]


def answer_valid(r: Record) -> bool:
    """Same validity condition as baselines.score_rule_prediction."""
    supp, sgns = r.get("pred_support") or [], r.get("pred_signs") or []
    n = int(r.get("n", 40))
    return len(supp) == len(sgns) and len(supp) > 0 and all(0 <= int(i) < n for i in supp)


def predict(inst: Any, r: Record, X: np.ndarray) -> np.ndarray:
    """Evaluate the answered rule exactly as baselines.score_rule_prediction does."""
    supp = [int(i) for i in r["pred_support"]]
    sgns = [1 if int(s) >= 0 else -1 for s in r["pred_signs"]]
    if inst.task_type in ("dtree2", "dlist3") and len(supp) != 3:
        # score_rule_prediction falls through to the majority form in this case.
        s = np.zeros(X.shape[0], dtype=np.int32)
        for i, g in zip(supp, sgns):
            s += g * X[:, i]
        return (s > 0).astype(np.int8)
    return inst._eval_rule(X, tuple(supp), tuple(sgns))


def verify_logged_scores(rows: Sequence[Record]) -> Dict[str, int]:
    """Regenerate every Exp 2 instance, check logged truth/decoy, and rescore the
    logged answer with baselines.score_rule_prediction; all must match."""
    from .baselines import score_rule_prediction

    n_checked = 0
    for r in rows:
        inst = instance_for(r)
        assert list(inst.true_support) == list(r["true_support"]), r["run_key"]
        assert list(inst.true_signs) == list(r["true_signs"]), r["run_key"]
        assert list(inst.decoy_support) == list(r["decoy_support"]), r["run_key"]
        m = score_rule_prediction(inst, r.get("pred_support") or [], r.get("pred_signs") or [])
        assert abs(m["heldout_acc"] - r["heldout_acc"]) < 1e-12, r["run_key"]
        assert m["matches_decoy"] == r["matches_decoy"], r["run_key"]
        if answer_valid(r):
            # Our evaluator must reproduce the scorer on the held-out points.
            yhat = predict(inst, r, inst.test_X)
            assert abs(float(np.mean(yhat == inst.test_y)) - r["heldout_acc"]) < 1e-12, r["run_key"]
        n_checked += 1
    n_test = {int(instance_for(r).test_X.shape[0]) for r in rows}
    assert len(n_test) == 1
    return {"n_rescored": n_checked, "n_heldout_points": n_test.pop()}


def arbitrate(bb: Record, wb: Record, k: int) -> Dict[str, Any]:
    """Keep the WB answer iff it is valid and agrees with the oracle on the first k check points."""
    if answer_valid(wb):
        X, y = check_points(wb)
        keep = bool(np.all(predict(instance_for(wb), wb, X[:k]) == y[:k]))
    else:
        keep = False
    src = wb if keep else bb
    return {"keep_wb": keep, "acc": float(src["heldout_acc"]), "trapped": trapped(src)}


def build_bb_wb_pairs(rows: Sequence[Record]) -> Dict[str, List[Tuple[Cell, Record, Record]]]:
    bb: Dict[Tuple[Cell, int], Record] = {}
    for r in rows:
        if r["mode"] == "BB":
            bb[(cell_key(r), int(r["seed"]))] = r
    out: Dict[str, List[Tuple[Cell, Record, Record]]] = defaultdict(list)
    for r in rows:
        if r["mode"] != "WB" or r["variant"] not in WB_VARIANTS:
            continue
        k = (cell_key(r), int(r["seed"]))
        if k in bb:
            out[r["variant"]].append((k[0], bb[k], r))
    for v in out:
        out[v].sort(key=lambda t: (t[0], int(t[2]["seed"])))
    return out


def arb_summary(pairs: Sequence[Tuple[Cell, Record, Record]], k: int, seed: int) -> Dict[str, Any]:
    by_cell: Dict[Cell, List[float]] = defaultdict(list)
    by_cell_wb: Dict[Cell, List[float]] = defaultdict(list)
    bb_acc, wb_acc, arb_acc, arb_trap, wb_trap = [], [], [], [], []
    n_trapped = n_trapped_rej = n_correct = n_correct_rej = n_keep = n_invalid = 0
    for c, b, w in pairs:
        a = arbitrate(b, w, k)
        bb_acc.append(b["heldout_acc"])
        wb_acc.append(w["heldout_acc"])
        arb_acc.append(a["acc"])
        arb_trap.append(a["trapped"])
        wb_trap.append(trapped(w))
        by_cell[c].append(a["acc"] - b["heldout_acc"])
        by_cell_wb[c].append(a["acc"] - w["heldout_acc"])
        n_keep += int(a["keep_wb"])
        n_invalid += int(not answer_valid(w))
        if trapped(w):
            n_trapped += 1
            n_trapped_rej += int(not a["keep_wb"])
        if w.get("exact_rule"):
            n_correct += 1
            n_correct_rej += int(not a["keep_wb"])
    pb = pooled_paired([np.array(v) for v in by_cell.values()], seed=seed)
    pw = pooled_paired([np.array(v) for v in by_cell_wb.values()], seed=seed + 1)
    return {
        "k": k,
        "n_pairs": len(pairs),
        "n_cells": len(by_cell),
        "acc_bb": float(np.mean(bb_acc)),
        "acc_wb": float(np.mean(wb_acc)),
        "acc_arb": float(np.mean(arb_acc)),
        "arb_minus_bb": pb["mean_diff"],
        "arb_minus_bb_ci95_strat": pb["strat_ci95"],
        "arb_minus_bb_ci95_cluster": pb["cluster_ci95"],
        "arb_minus_wb": pw["mean_diff"],
        "arb_minus_wb_ci95_strat": pw["strat_ci95"],
        "share_wb_kept": n_keep / len(pairs),
        "n_wb_invalid": n_invalid,
        "n_wb_trapped": n_trapped,
        "share_trapped_rejected": (n_trapped_rej / n_trapped) if n_trapped else None,
        "n_wb_correct": n_correct,
        "n_wb_correct_rejected": n_correct_rej,
        "trap_rate_wb": float(np.mean(wb_trap)),
        "trap_rate_arb": float(np.mean(arb_trap)),
    }


def part3(rows: Sequence[Record]) -> Dict[str, Any]:
    ver = verify_logged_scores(rows)
    pairs = build_bb_wb_pairs(rows)
    wb_total = {v: sum(1 for r in rows if r["mode"] == "WB" and r["variant"] == v) for v in WB_VARIANTS}
    unpaired = {v: wb_total[v] - len(pairs.get(v, [])) for v in WB_VARIANTS}
    res: Dict[str, Any] = {"verification": ver, "wb_total": wb_total, "wb_unpaired": unpaired, "main": {}, "all_cells": {}}
    for vi, v in enumerate(WB_VARIANTS):
        ps = pairs.get(v, [])
        main_ps = [t for t in ps if is_main(t[2])]
        for scope, sel in (("main", main_ps), ("all_cells", ps)):
            res[scope][v] = [arb_summary(sel, k, seed=SEED + 300 + 10 * vi + ki) for ki, k in enumerate(ARB_KS)]
    code_main = [t for t in pairs.get(CODE, []) if is_main(t[2])]
    assert len(code_main) == EXPECTED_MAIN_PER_ARM, len(code_main)
    per_model: Dict[str, Any] = {}
    for m in sorted({t[2]["model"] for t in code_main}):
        sel = [t for t in code_main if t[2]["model"] == m]
        per_model[m] = {str(k): arb_summary(sel, k, seed=SEED + 400 + k) for k in ARB_KS}
    res["code_main_per_model"] = per_model
    # Resource use of the combined system (logged per run).
    def mean_of(field: str, side: int, sel: Sequence[Tuple[Cell, Record, Record]]) -> float:
        return float(np.mean([float(t[side].get(field, 0.0)) for t in sel]))

    res["resources_code_main"] = {
        f: {"bb": mean_of(f, 1, code_main), "wb": mean_of(f, 2, code_main)}
        for f in ("queries_used", "turns", "output_tokens", "cost_usd")
    }
    return res


# ==============================================================================
# REPORTING
# ==============================================================================
def f3(x: Optional[float]) -> str:
    return "  n/a" if x is None else f"{x:+.3f}"


def ci(t: Sequence[float]) -> str:
    return f"[{t[0]:+.3f}, {t[1]:+.3f}]"


def print_pooled(title: str, p: Dict[str, Any]) -> None:
    print(f"  {title}")
    print(f"    cells={p['n_cells']} pairs={p['n_pairs']} mean diff={p['mean_diff']:+.4f} (mean of cell means {p['mean_of_cell_means']:+.4f})")
    print(f"    stratified bootstrap (cells fixed): 90% {ci(p['strat_ci90'])}  95% {ci(p['strat_ci95'])}")
    print(f"    cluster bootstrap (cells resampled): 90% {ci(p['cluster_ci90'])}  95% {ci(p['cluster_ci95'])}")
    print(f"    sign-flip permutation p (two-sided) = {p['signflip_p_two_sided']:.4f}")
    for d, v in p["equivalence"].items():
        print(f"    equivalent at margin {d}: stratified={'YES' if v['stratified'] else 'no'}  cluster={'YES' if v['cluster'] else 'no'}")


def print_cells(cells: Sequence[Dict[str, Any]], la: str, lb: str) -> None:
    print(f"    {'model':26s} {'effort':7s} {'track':22s} {'task':9s} {'mt':>4s} {'n':>3s} {la:>7s} {lb:>7s} {'diff':>7s}  95% CI")
    for c in cells:
        m, e, tr, tk, mt = c["cell"]
        print(
            f"    {m:26s} {e:7s} {tr:22s} {tk:9s} {mt:>4d} {c['n_pairs']:>3d} {c['mean_a']:7.3f} {c['mean_b']:7.3f} {c['diff']:+7.3f}  {ci(c['ci95'])}"
        )


def report_text(p1: Dict[str, Any], p2: Dict[str, Any], n_records: int) -> None:
    print("=== MECHANISM ANALYSIS: EXPERIMENT 2 ===")
    print(f"Exp 2 records loaded: {n_records} (from {', '.join(s.name for s in SOURCES)})")
    print(f"Bootstrap draws={N_BOOT}, permutation draws={N_PERM}, base seed={SEED}. Acc of an invalid answer = 0.50.")

    print("\n##### PART 1: anti-transparent code vs refutable control #####")
    a = p1["acc_code_minus_refut"]
    print("\n[1a] Per-cell paired Acc (code minus refutable):")
    print_cells(a["cells"], "code", "refut")
    print()
    print_pooled("[1b] Pooled Acc difference, code minus refutable:", a["pooled"])
    print()
    print_pooled("[1c] Pooled trap-rate difference, code minus refutable:", p1["trap_code_minus_refut"]["pooled"])
    print("\n[1d] Sensitivity contrast (same 18 cells): refutable minus short control, per cell:")
    print_cells(p1["acc_refut_minus_short"]["cells"], "refut", "short")
    print()
    print_pooled("Pooled Acc difference, refutable minus short:", p1["acc_refut_minus_short"]["pooled"])
    print()
    print_pooled("Pooled trap-rate difference, refutable minus short:", p1["trap_refut_minus_short"]["pooled"])
    print()
    pf = p1["parse_failures_in_pairs"]
    print(f"[1e] Invalid answers among the 405 pairs: code={pf['code_invalid']} refut={pf['refut_invalid']} pairs with either invalid={pf['pairs_any_invalid']}")
    print_pooled("Pooled Acc difference, code minus refutable, both answers valid JSON:", p1["acc_code_minus_refut_valid_json_only"]["pooled"])

    print("\n##### PART 2: what the agents did #####")
    print(f"\nMain condition (matched_medium_budget, majority3, effort medium/default, no max_tokens override): counts per arm {p2['main_counts']}")
    print(f"Models ({len(p2['main_models'])}); refutable control present for {len(p2['refut_models_main'])}: {', '.join(p2['refut_models_main'])}")
    print("\n[2a] Per model x arm. q=mean queries, med=median queries, q0=share with zero oracle queries, py=mean python calls,")
    print("     turns=mean turns, trap=matches_decoy rate (BB: chance baseline), acc=mean Acc, json=valid-JSON rate")
    print(f"    {'model':26s} {'arm':7s} {'n':>3s} {'q':>6s} {'med':>5s} {'q0':>5s} {'py':>5s} {'turns':>5s} {'trap':>5s} {'acc':>5s} {'json':>5s}")
    for m, arms in p2["table"].items():
        for ar in ARMS:
            if ar not in arms:
                continue
            d = arms[ar]
            print(
                f"    {m:26s} {ARM_SHORT[ar]:7s} {d['n']:>3d} {d['mean_queries']:6.1f} {d['median_queries']:5.1f} {d['zero_query_share']:5.2f} "
                f"{d['mean_python_calls']:5.2f} {d['mean_turns']:5.2f} {d['trap_rate']:5.2f} {d['mean_acc']:5.3f} {d['valid_json_rate']:5.2f}"
            )
        print()

    def routes(title: str, rr: Dict[str, Any]) -> None:
        print(f"  {title} (n={rr['n']})")
        for g, v in rr["groups"].items():
            acc_s = "  n/a" if v["mean_acc"] is None else f"{v['mean_acc']:.3f}"
            ex_s = "  n/a" if v["exact_rule_rate"] is None else f"{v['exact_rule_rate']:.2f}"
            print(f"    {g:18s} n={v['n']:3d} share={v['share']:.3f} acc={acc_s} exact_rule={ex_s}")
        z, s = rr["zero_oracle_queries"], rr["some_oracle_queries"]
        za = "n/a" if z["mean_acc"] is None else f"{z['mean_acc']:.3f}"
        sa = "n/a" if s["mean_acc"] is None else f"{s['mean_acc']:.3f}"
        print(f"    queries_used==0: n={z['n']} share={z['share']:.3f} acc={za};  queries_used>0: n={s['n']} share={s['share']:.3f} acc={sa}")

    print("[2b] Routes to the answer, main condition (by tools attempted in tool_sequence):")
    routes("Opaque weights (wb_inv_relu)", p2["relu_routes_main"])
    routes("Anti-transparent code (wb_inv_code), for comparison", p2["code_routes_main"])
    routes("Black-box, for comparison", p2["bb_routes_main"])
    print("  Opaque weights per model: share with zero oracle queries / share Python-only / Acc(zero-query) / Acc(queried)")
    for m, rr in p2["relu_routes_by_model"].items():
        z, s = rr["zero_oracle_queries"], rr["some_oracle_queries"]
        print(
            f"    {m:26s} n={rr['n']:2d} q0={z['share']:.2f} py_only={rr['groups']['python_only']['share']:.2f} "
            f"acc_q0={'  n/a' if z['mean_acc'] is None else format(z['mean_acc'], '.3f')} "
            f"acc_q>0={'  n/a' if s['mean_acc'] is None else format(s['mean_acc'], '.3f')}"
        )

    def tt(title: str, res: Dict[str, Any]) -> None:
        print(f"\n  {title}")
        print(
            f"    strata eligible (>= {res['min_each']} trapped and >= {res['min_each']} untrapped): {res['n_strata_eligible']} of {res['n_strata_total']}; "
            f"runs used: trapped={res['n_trapped_eligible']} untrapped={res['n_untrapped_eligible']} "
            f"(all runs: trapped={res['n_trapped_all']} untrapped={res['n_untrapped_all']})"
        )
        print("    per stratum: n_trap n_untrap | mean q trapped / untrapped | zero-q share trapped / untrapped")
        for s in res["per_stratum"]:
            lab = s["stratum"] if isinstance(s["stratum"], str) else "|".join(str(x) for x in s["stratum"])
            print(
                f"      {lab:60s} {s['n_trapped']:3d} {s['n_untrapped']:3d} | {s['mean_q_trapped']:6.1f} / {s['mean_q_untrapped']:6.1f} | "
                f"{s['zero_q_trapped']:.2f} / {s['zero_q_untrapped']:.2f}"
            )
        print("    pooled trapped minus untrapped (CMH-weighted within strata; naive = unstratified):")
        for name, v in res["metrics"].items():
            print(
                f"      {name:27s} strata={v['n_strata']:2d} n={v['n_a']:3d}/{v['n_b']:3d} trapped={v['mean_a']:9.3f} untrapped={v['mean_b']:9.3f} "
                f"naive={v['naive_diff']:+9.3f} stratified={v['strat_diff']:+9.3f} 95% CI {ci(v['ci95'])} perm p={v['perm_p_two_sided']:.4f}"
            )

    print("\n[2c] Anti-transparent code: trapped vs untrapped")
    tt("Main condition, stratified by model:", p2["trapped_vs_untrapped_main"])
    tt("All Exp 2 cells, stratified by (model, effort, budget_track, task_type, max_tokens_override):", p2["trapped_vs_untrapped_all_cells"])

    print("\n[2d] Ordering (anti-transparent code): oracle call listed after the first run_python call")
    for lab, od in (("main", p2["ordering_main"]), ("all cells", p2["ordering_all_cells"])):
        print(
            f"  {lab}: runs={od['n_runs']} with python={od['n_with_python']} oracle-after-first-python={od['n_oracle_after_first_python']}; "
            f"trapped {f3(od['trapped']['share_oracle_after_first_python'])} (n={od['trapped']['n_with_python']}), "
            f"untrapped {f3(od['untrapped']['share_oracle_after_first_python'])} (n={od['untrapped']['n_with_python']})"
        )
        print(
            f"    share of runs whose tool calls all fit in one assistant turn (turns<=2): all={f3(od['share_single_tool_batch_all'])}, "
            f"among oracle-after-python runs={f3(od['share_single_tool_batch_among_after'])}"
        )
    print("  (model-stratified version: metric 'oracle_after_first_python' in [2c])")
    print("  Caveat: tool_sequence is a flat list without turn boundaries or call arguments; calls issued in the")
    print("  same assistant turn are listed in emission order but were all issued before any result was seen.")

    for lab, tw in (("main", p2["trapped_with_queries_main"]), ("all cells", p2["trapped_with_queries_all_cells"])):
        print(
            f"\n[2e] Trapped code runs that queried the oracle ({lab}): {tw['n_trapped_with_queries']} of {tw['n_trapped']} "
            f"(>= 10 queries: {tw['n_trapped_with_ge10_queries']})"
        )


def report_part3(p3: Dict[str, Any]) -> None:
    ver = p3["verification"]
    print("\n##### PART 3: offline replay of verification-first arbitration #####")
    print(
        f"Instances regenerated with generators.generate_blackwell_inv_instance; all {ver['n_rescored']} logged answers rescored "
        f"with baselines.score_rule_prediction and matched the logged heldout_acc / matches_decoy / truth / decoy."
    )
    print(f"Held-out scoring set: {ver['n_heldout_points']} points per instance.")
    print("Rule: draw k fresh uniform inputs (separate RNG stream from the held-out points; first k of 20, so checks are nested),")
    print("label them with the oracle function (eval_inv), keep the WB answer iff it is valid and agrees on all k;")
    print("otherwise use the BB answer for the same (model, effort, budget_track, task_type, max_tokens_override, seed).")
    print(f"WB runs per variant: {p3['wb_total']}; without a same-key BB partner (excluded): {p3['wb_unpaired']}")
    for scope, title in (("main", "Main condition"), ("all_cells", "All Exp 2 cells pooled")):
        print(f"\n  {title}:")
        print(
            f"    {'variant':30s} {'k':>3s} {'pairs':>5s} {'BB':>6s} {'WB':>6s} {'arb':>6s} {'arb-BB':>7s} {'95% CI (strat)':>18s} "
            f"{'arb-WB':>7s} {'kept':>5s} {'trapRej':>8s} {'corrRej':>8s} {'trapWB':>6s} {'trapArb':>7s}"
        )
        for v, lst in p3[scope].items():
            for s in lst:
                tr = "   n/a" if s["share_trapped_rejected"] is None else f"{s['share_trapped_rejected']:.3f}"
                print(
                    f"    {v:30s} {s['k']:>3d} {s['n_pairs']:>5d} {s['acc_bb']:6.3f} {s['acc_wb']:6.3f} {s['acc_arb']:6.3f} "
                    f"{s['arb_minus_bb']:+7.3f} {ci(s['arb_minus_bb_ci95_strat']):>18s} {s['arb_minus_wb']:+7.3f} {s['share_wb_kept']:5.2f} "
                    f"{tr:>4s}({s['n_wb_trapped']:3d}) {s['n_wb_correct_rejected']:3d}/{s['n_wb_correct']:3d} {s['trap_rate_wb']:6.3f} {s['trap_rate_arb']:7.3f}"
                )
    print("    (trapRej = share of trapped WB answers rejected (n trapped); corrRej = exactly correct WB answers rejected / total)")
    print("\n  Anti-transparent code, main condition, per model (k = 5 / 10 / 20):")
    print(f"    {'model':26s} {'n':>3s} {'BB':>6s} {'WB':>6s} {'arb5':>6s} {'arb10':>6s} {'arb20':>6s} {'arb10-BB':>9s} {'trapWB':>6s} {'trapArb10':>9s}")
    for m, d in p3["code_main_per_model"].items():
        s5, s10, s20 = d["5"], d["10"], d["20"]
        print(
            f"    {m:26s} {s10['n_pairs']:>3d} {s10['acc_bb']:6.3f} {s10['acc_wb']:6.3f} {s5['acc_arb']:6.3f} {s10['acc_arb']:6.3f} "
            f"{s20['acc_arb']:6.3f} {s10['arb_minus_bb']:+9.3f} {s10['trap_rate_wb']:6.3f} {s10['trap_rate_arb']:9.3f}"
        )
    rc = p3["resources_code_main"]
    print("\n  Budget caveat. The arbitrated system consumes one complete BB run plus one complete WB run plus k extra")
    print("  oracle queries, and is compared here against single-run baselines (BB alone, WB alone) that each had one")
    print("  budget. Mean per-run use in the main condition (anti-transparent code pairs): "
          + ", ".join(f"{f} BB={v['bb']:.2f} WB={v['wb']:.2f}" for f, v in rc.items()))
    print("  What the logs can say: given two independently logged runs, how often a k-point oracle check separates a")
    print("  trapped WB answer from a correct one, and what the combined answer would score. What they cannot say: how one")
    print("  agent given split budgets (or the same total budget) would behave, whether a single BB run at double budget")
    print("  would match the arbitrated accuracy (no such runs at equal total budget per seed), or how agents would act if")
    print("  told a verification step follows. The k check queries are free here (computed offline), not drawn from either budget.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-json", default=str(ROOT / "report_mechanism.json"))
    args = ap.parse_args(argv)

    rows = load_exp2()
    assert len(rows) == EXPECTED_EXP2_RECORDS, f"expected {EXPECTED_EXP2_RECORDS} Exp 2 records, got {len(rows)}"
    assert len({r["run_key"] for r in rows}) == len(rows), "duplicate run_key across sources"
    # queries_used > 0 must coincide with at least one successful oracle call in tool_sequence.
    for r in rows:
        ok_q = any(t.startswith("query_oracle") and t.endswith(":ok") for t in r["tool_sequence"])
        assert ok_q == (r["queries_used"] > 0), r["run_key"]
        assert python_calls_seq(r) == r["python_calls"], r["run_key"]

    p1 = part1(rows)
    p2 = part2(rows)
    p3 = part3(rows)
    report = {
        "n_exp2_records": len(rows),
        "n_boot": N_BOOT,
        "n_perm": N_PERM,
        "seed": SEED,
        "part1": p1,
        "part2": p2,
        "part3": p3,
    }
    Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_text(p1, p2, len(rows))
    report_part3(p3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
