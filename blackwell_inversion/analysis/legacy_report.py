"""The stage report of the original analysis (H1-H4 verdicts per hypothesis of the protocol).

It regenerates `reports/legacy/report_master_final.{json,md}` byte-identically from
`data/main/results_c2_pilot.jsonl`:

    python -m blackwell_inversion.analysis.legacy_report data/main/results_c2_pilot.jsonl \
        --json-out report_master_final.json --md-out report_master_final.md

The two earlier stage reports (`report_c1_haiku55.*`, `report_c2_step1.*`) were written
by earlier versions of this code and are kept for provenance only.

Exp 2 comparisons also report the equivalent black-box query count q_eq (the number of
random queries at which a correlation learner matches the WB accuracy) and
Gamma_WB = log2(q_eq / q_B).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..baselines import compute_bb_query_curve, interpolate_q_eq
from .stats import fisher_adv_test, holm_bonferroni, load_jsonl, mcnemar_exact, paired_delta_stats


def analyze_exp1(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    exp1 = [r for r in records if r.get("experiment") == "exp1_vbb"]
    groups: Dict[Tuple[str, str, str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in exp1:
        key = (
            str(r["model"]),
            str(r.get("effort", "default")),
            str(r["family"]),
            str(r["track"]),
            str(r["regime"]),
            str(r["mode"]),
            str(r["substrate"]),
        )
        groups[key].append(r)

    cells: List[Dict[str, Any]] = []
    for (model, effort, family, track, regime, mode, sub), rows in sorted(groups.items()):
        pos = [r for r in rows if int(r["planted"]) == 1]
        neg = [r for r in rows if int(r["planted"]) == 0]
        tp = sum(1 for r in pos if r.get("decision_int") == 1)
        fp = sum(1 for r in neg if r.get("decision_int") == 1)
        parse_rate = float(np.mean([1.0 if r.get("parse_ok") else 0.0 for r in rows])) if rows else 0.0
        mean_think = float(np.mean([float(r.get("thinking_tokens", 0)) for r in rows])) if rows else 0.0
        trunc_cnt = int(sum(int(r.get("truncated_turns", 0)) for r in rows))
        alt = "two-sided" if regime == "control_impossible" else "greater"
        st = fisher_adv_test(tp, len(pos), fp, len(neg), alternative=alt)
        cells.append(
            {
                "model": model,
                "effort": effort,
                "family": family,
                "track": track,
                "regime": regime,
                "mode": mode,
                "substrate": sub,
                "n_total": len(rows),
                "parse_rate": parse_rate,
                "mean_thinking_tokens": round(mean_think, 1),
                "truncated_turns": trunc_cnt,
                "mean_cost_usd": float(np.mean([float(r.get("cost_usd", 0.0)) for r in rows])),
                **st,
            }
        )

    # Compute Delta_WB relative to matching (model, effort, family, track, regime) BB cell
    bb_lookup: Dict[Tuple[str, str, str, str, str], float] = {
        (c["model"], c["effort"], c["family"], c["track"], c["regime"]): c["adv"]
        for c in cells
        if c["mode"] == "BB"
    }
    for c in cells:
        bb_adv = bb_lookup.get((c["model"], c["effort"], c["family"], c["track"], c["regime"]), 0.0)
        c["delta_wb"] = float(c["adv"] - bb_adv) if c["mode"] == "WB" else 0.0

    # Holm-Bonferroni adjustment across all WB non-control cells (global + per-family)
    test_indices = [i for i, c in enumerate(cells) if c["mode"] == "WB" and c["regime"] != "control_impossible"]
    adj_ps = holm_bonferroni([cells[i]["p_value"] for i in test_indices])
    for idx, p_adj in zip(test_indices, adj_ps):
        cells[idx]["p_holm"] = p_adj

    # Per-family Holm-Bonferroni + H1 verdict per instance family
    by_family_idx: Dict[str, List[int]] = defaultdict(list)
    for idx in test_indices:
        by_family_idx[cells[idx]["family"]].append(idx)
    for idxs in by_family_idx.values():
        fam_adj = holm_bonferroni([cells[i]["p_value"] for i in idxs])
        for idx, p_adj in zip(idxs, fam_adj):
            cells[idx]["p_holm_family"] = p_adj

    # Control leak gate check
    ctrl_cells = [c for c in cells if c["regime"] == "control_impossible"]
    ctrl_ps = holm_bonferroni([c["p_value"] for c in ctrl_cells])
    leak_failures: List[Dict[str, Any]] = []
    for c, p_adj in zip(ctrl_cells, ctrl_ps):
        c["p_holm"] = p_adj
        c["p_holm_family"] = p_adj
        if abs(c["adv"]) >= 0.10 and p_adj < 0.05:
            leak_failures.append(c)

    # H1 verdict: vbb_hard near 0 while transparent > 0 (Holm p < 0.05)
    vbb_wb = [c for c in cells if c["mode"] == "WB" and c["regime"] == "vbb_hard"]
    trans_wb = [c for c in cells if c["mode"] == "WB" and c["regime"] == "transparent"]
    vbb_breached = [c for c in vbb_wb if c["adv"] >= 0.15 and c.get("p_holm", 1.0) < 0.05]
    trans_solved = [c for c in trans_wb if c["adv"] >= 0.30 and c.get("p_holm", 1.0) < 0.05]

    if leak_failures:
        h1_verdict = "INVALID_LEAK_DETECTED"
    elif not vbb_wb or not trans_wb:
        h1_verdict = "INCONCLUSIVE_MISSING_CELLS"
    elif not vbb_breached and trans_solved:
        h1_verdict = "SUPPORTED"
    elif vbb_breached:
        h1_verdict = "REFUTED_VBB_BREACHED"
    else:
        h1_verdict = "INCONCLUSIVE_TRANSPARENT_UNSOLVED"

    h1_verdict_by_family: Dict[str, str] = {}
    all_families = sorted({c["family"] for c in cells})
    for fam in all_families:
        f_vbb = [c for c in cells if c["family"] == fam and c["mode"] == "WB" and c["regime"] == "vbb_hard"]
        f_trans = [c for c in cells if c["family"] == fam and c["mode"] == "WB" and c["regime"] == "transparent"]
        f_breached = [c for c in f_vbb if c["adv"] >= 0.15 and c.get("p_holm_family", 1.0) < 0.05]
        f_solved = [c for c in f_trans if c["adv"] >= 0.30 and c.get("p_holm_family", 1.0) < 0.05]
        if any(lf["family"] == fam for lf in leak_failures):
            h1_verdict_by_family[fam] = "INVALID_LEAK_DETECTED"
        elif not f_vbb or not f_trans:
            h1_verdict_by_family[fam] = "INCONCLUSIVE_MISSING_CELLS"
        elif not f_breached and f_solved:
            h1_verdict_by_family[fam] = "SUPPORTED"
        elif f_breached:
            h1_verdict_by_family[fam] = "REFUTED_VBB_BREACHED"
        else:
            h1_verdict_by_family[fam] = "INCONCLUSIVE_TRANSPARENT_UNSOLVED"

    # H3 substrate ordering check on transparent WB cells per (model, effort, family, track)
    h3_checks: List[Dict[str, Any]] = []
    sub_map: Dict[Tuple[str, str, str, str], Dict[str, float]] = defaultdict(dict)
    for c in trans_wb:
        sub_map[(c["model"], c["effort"], c["family"], c["track"])][c["substrate"]] = c["delta_wb"]
    for (model, effort, family, track), sm in sorted(sub_map.items()):
        if {"code_clean", "code_obf", "relu_weights"}.issubset(sm.keys()):
            ordered = sm["code_clean"] >= sm["code_obf"] >= sm["relu_weights"]
            strict_drop = sm["code_clean"] > sm["relu_weights"] + 0.05
            h3_checks.append(
                {
                    "model": model,
                    "effort": effort,
                    "family": family,
                    "track": track,
                    "code_clean": sm["code_clean"],
                    "code_obf": sm["code_obf"],
                    "relu_weights": sm["relu_weights"],
                    "relu_weights_dense": sm.get("relu_weights_dense"),
                    "monotone_nonincreasing": ordered,
                    "strict_code_over_relu": strict_drop,
                }
            )

    return {
        "n_records": len(exp1),
        "leak_gate_passed": len(leak_failures) == 0,
        "leak_failures": leak_failures,
        "cells": cells,
        "h1_verdict": h1_verdict,
        "h1_verdict_by_family": h1_verdict_by_family,
        "h3_substrate_checks": h3_checks,
    }


def analyze_exp2(
    records: Sequence[Dict[str, Any]], bb_curve: Optional[Dict[int, Dict[str, float]]] = None
) -> Dict[str, Any]:
    exp2 = [r for r in records if r.get("experiment") == "exp2_blackwell"]
    if not exp2:
        return {"n_records": 0, "comparisons": [], "h2_verdict": "NO_DATA", "h4_scaling": []}

    bb_curves_by_task: Dict[str, Dict[int, Dict[str, float]]] = {}
    if bb_curve is not None:
        bb_curves_by_task["majority3"] = bb_curve

    def _get_bb_curve(tt: str) -> Dict[int, Dict[str, float]]:
        if tt not in bb_curves_by_task:
            bb_curves_by_task[tt] = compute_bb_query_curve(
                n=40, support_size=3, n_trials=200, seed=42, task_type=tt
            )
        return bb_curves_by_task[tt]

    # Index BB baselines by (model, effort, task_type, budget_track, seed)
    bb_by_seed: Dict[Tuple[str, str, str, str, int], Dict[str, Any]] = {}
    wb_groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)

    for r in exp2:
        eff = str(r.get("effort", "default"))
        tt = str(r.get("task_type", "majority3"))
        key_s = (str(r["model"]), eff, tt, str(r["budget_track"]), int(r["seed"]))
        if r.get("mode") == "BB":
            bb_by_seed[key_s] = r
        else:
            wb_groups[(str(r["model"]), eff, tt, str(r["budget_track"]), str(r["variant"]))].append(r)

    comparisons: List[Dict[str, Any]] = []
    for (model, effort, task_type, track, variant), wb_rows in sorted(wb_groups.items()):
        wb_accs: List[float] = []
        bb_accs: List[float] = []
        wb_exacts: List[int] = []
        bb_exacts: List[int] = []
        decoy_hits: List[float] = []
        wb_queries: List[int] = []
        bb_queries: List[int] = []
        wb_turns: List[int] = []
        bb_turns: List[int] = []
        wb_think: List[float] = []
        bb_think: List[float] = []
        wb_parse: List[float] = []
        bb_parse: List[float] = []
        trunc_total = 0
        q_max_ref = 80

        for r_wb in sorted(wb_rows, key=lambda x: int(x["seed"])):
            key_s = (model, effort, task_type, track, int(r_wb["seed"]))
            if key_s not in bb_by_seed:
                continue
            r_bb = bb_by_seed[key_s]
            wb_accs.append(float(r_wb.get("heldout_acc", 0.5)))
            bb_accs.append(float(r_bb.get("heldout_acc", 0.5)))
            wb_exacts.append(int(round(float(r_wb.get("exact_rule", 0.0)))))
            bb_exacts.append(int(round(float(r_bb.get("exact_rule", 0.0)))))
            decoy_hits.append(float(r_wb.get("matches_decoy", 0.0)))
            wb_queries.append(int(r_wb.get("queries_used", 0)))
            bb_queries.append(int(r_bb.get("queries_used", 0)))
            wb_turns.append(int(r_wb.get("turns", 0)))
            bb_turns.append(int(r_bb.get("turns", 0)))
            wb_think.append(float(r_wb.get("thinking_tokens", 0)))
            bb_think.append(float(r_bb.get("thinking_tokens", 0)))
            wb_parse.append(1.0 if r_wb.get("parse_ok") else 0.0)
            bb_parse.append(1.0 if r_bb.get("parse_ok") else 0.0)
            trunc_total += int(r_wb.get("truncated_turns", 0)) + int(r_bb.get("truncated_turns", 0))
            q_max_ref = int(r_bb.get("q_max", q_max_ref))

        if variant == "wb_clean":
            alt = "greater"
        elif variant.startswith("wb_inv_"):
            alt = "less"
        else:
            alt = "two-sided"
        pstats = paired_delta_stats(wb_accs, bb_accs, alternative=alt, seed=2026)
        mc = mcnemar_exact(wb_exacts, bb_exacts, alternative=alt)

        task_curve = _get_bb_curve(task_type)
        q_eq = interpolate_q_eq(pstats["mean_wb"], task_curve, metric_key="mean_acc")
        q_b_eff = max(float(np.mean(bb_queries)) if bb_queries else float(q_max_ref), 1.0)
        gamma_wb = math.log2(max(q_eq, 0.5) / q_b_eff) if pstats["n_pairs"] > 0 else 0.0

        comparisons.append(
            {
                "model": model,
                "effort": effort,
                "task_type": task_type,
                "budget_track": track,
                "variant": variant,
                "n_pairs": pstats["n_pairs"],
                "mean_bb_acc": pstats["mean_bb"],
                "mean_wb_acc": pstats["mean_wb"],
                "delta_wb": pstats["delta_wb"],
                "ci_lo": pstats["ci_lo"],
                "ci_hi": pstats["ci_hi"],
                "p_value": pstats["p_value"],
                "mean_bb_exact": float(np.mean(bb_exacts)) if bb_exacts else 0.0,
                "mean_wb_exact": float(np.mean(wb_exacts)) if wb_exacts else 0.0,
                "mcnemar_p": mc["p_value"],
                "decoy_trap_rate": float(np.mean(decoy_hits)) if decoy_hits else 0.0,
                "mean_bb_queries": float(np.mean(bb_queries)) if bb_queries else 0.0,
                "mean_wb_queries": float(np.mean(wb_queries)) if wb_queries else 0.0,
                "mean_bb_turns": round(float(np.mean(bb_turns)), 2) if bb_turns else 0.0,
                "mean_wb_turns": round(float(np.mean(wb_turns)), 2) if wb_turns else 0.0,
                "bb_parse_rate": float(np.mean(bb_parse)) if bb_parse else 0.0,
                "wb_parse_rate": float(np.mean(wb_parse)) if wb_parse else 0.0,
                "mean_bb_thinking_tokens": round(float(np.mean(bb_think)), 1) if bb_think else 0.0,
                "mean_wb_thinking_tokens": round(float(np.mean(wb_think)), 1) if wb_think else 0.0,
                "truncated_turns": trunc_total,
                "q_eq": round(q_eq, 2),
                "gamma_wb_bits": round(gamma_wb, 3),
            }
        )

    # Holm-Bonferroni across all anti-transparent (wb_inv_*) comparisons (global + per-cell)
    inv_idx = [i for i, c in enumerate(comparisons) if c["variant"].startswith("wb_inv_")]
    adj_ps = holm_bonferroni([comparisons[i]["p_value"] for i in inv_idx])
    for idx, p_adj in zip(inv_idx, adj_ps):
        comparisons[idx]["p_holm"] = p_adj

    by_cell_idx: Dict[Tuple[str, str, str, str], List[int]] = defaultdict(list)
    for idx in inv_idx:
        c = comparisons[idx]
        by_cell_idx[(c["model"], c["effort"], c["task_type"], c["budget_track"])].append(idx)
    for _, idxs in by_cell_idx.items():
        cell_adj = holm_bonferroni([comparisons[i]["p_value"] for i in idxs])
        for idx, p_adj in zip(idxs, cell_adj):
            comparisons[idx]["p_holm_cell"] = p_adj

    for c in comparisons:
        c.setdefault("p_holm", c["p_value"])
        c.setdefault("p_holm_cell", c["p_value"])

    # H2 verdict separately for Mech I+II (matched_total_budget / matched_medium_budget) and Pure Mech II
    mech1_hits = [
        c
        for c in comparisons
        if c["budget_track"] in ("matched_total_budget", "matched_medium_budget")
        and c["variant"].startswith("wb_inv_")
        and c["delta_wb"] <= -0.05
        and c["p_holm"] < 0.05
    ]
    mech2_hits = [
        c
        for c in comparisons
        if c["budget_track"] == "matched_query_free_code"
        and c["variant"].startswith("wb_inv_")
        and c["delta_wb"] <= -0.05
        and c["p_holm"] < 0.05
    ]
    if mech1_hits and mech2_hits:
        h2_verdict = "SUPPORTED_BOTH_MECH1_AND_MECH2"
    elif mech1_hits:
        h2_verdict = "SUPPORTED_MECH1_ONLY"
    elif mech2_hits:
        h2_verdict = "SUPPORTED_MECH2_ONLY"
    else:
        h2_verdict = "NOT_SUPPORTED"

    # H4 cross-model / cross-effort comparison table on C_inv
    h4_scaling: List[Dict[str, Any]] = []
    by_model: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for c in comparisons:
        if c["variant"].startswith("wb_inv_"):
            by_model[(c["model"], c["effort"])].append(c)
    for (model, effort), rows in sorted(by_model.items()):
        h4_scaling.append(
            {
                "model": model,
                "effort": effort,
                "mean_inv_delta_wb": float(np.mean([r["delta_wb"] for r in rows])),
                "mean_gamma_wb_bits": float(np.mean([r["gamma_wb_bits"] for r in rows])),
                "mean_decoy_trap_rate": float(np.mean([r["decoy_trap_rate"] for r in rows])),
                "mean_wb_thinking_tokens": float(np.mean([r["mean_wb_thinking_tokens"] for r in rows])),
            }
        )

    return {
        "n_records": len(exp2),
        "comparisons": comparisons,
        "h2_verdict": h2_verdict,
        "h4_scaling": h4_scaling,
    }


def format_markdown_report(res1: Dict[str, Any], res2: Dict[str, Any], total_cost: float) -> str:
    fam_verdicts = ", ".join(
        f"`{fam}={v}`" for fam, v in sorted(res1.get("h1_verdict_by_family", {}).items())
    )
    lines: List[str] = [
        "# Pilot Analysis Report — Observer-Relative VBB & Computational Blackwell Inversion",
        "",
        f"- **Total logged JSONL API spend:** `${total_cost:.4f}`",
        f"- **Exp 1 records:** `{res1['n_records']}` | **Leak Gate:** `{'PASS' if res1['leak_gate_passed'] else 'FAIL'}` | **H1 Verdict (pooled):** `{res1['h1_verdict']}`",
        f"- **H1 Verdict by Family:** {fam_verdicts if fam_verdicts else 'N/A'}",
        f"- **Exp 2 records:** `{res2['n_records']}` | **H2 Verdict:** `{res2['h2_verdict']}`",
        "",
    ]

    if res1["cells"]:
        lines.extend(
            [
                "## Experiment 1: Observer-Relative VBB (`C_kXOR`)",
                "",
                "| Model | Effort | Family | Track | Regime | Mode | Substrate | N | Parse | ThinkTok | Adv (95% CI) | Delta_WB | p_raw | p_holm_fam |",
                "| :--- | :---: | :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for c in res1["cells"]:
            pr = f"{c['p_value']:.4g}" if c["mode"] == "WB" or c["regime"] == "control_impossible" else "-"
            pf = f"{c['p_holm_family']:.4g}" if "p_holm_family" in c else "-"
            lines.append(
                f"| `{c['model']}` | {c['effort']} | `{c['family']}` | {c['track']} | {c['regime']} | {c['mode']} | {c['substrate']} | "
                f"{c['n_total']} | {c['parse_rate']:.2f} | {c['mean_thinking_tokens']:.0f} | "
                f"{c['adv']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] | "
                f"{c['delta_wb']:+.3f} | {pr} | {pf} |"
            )
        lines.append("")

    if res2["comparisons"]:
        lines.extend(
            [
                "## Experiment 2: Computational Blackwell Inversion (`C_inv`)",
                "",
                "| Model | Effort | Task | Budget Track | Variant | Pairs | Turns(B/W) | Parse(B/W) | Think(B/W) | BB Acc | WB Acc | Delta_WB (95% CI) | p_raw | p_holm_cell | p_holm_all | Gamma_WB (bits) | Decoy |",
                "| :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for c in res2["comparisons"]:
            lines.append(
                f"| `{c['model']}` | {c['effort']} | {c.get('task_type', 'majority3')} | {c['budget_track']} | {c['variant']} | {c['n_pairs']} | "
                f"{c.get('mean_bb_turns', 0):.2f}/{c.get('mean_wb_turns', 0):.2f} | "
                f"{c['bb_parse_rate']:.2f}/{c['wb_parse_rate']:.2f} | "
                f"{c['mean_bb_thinking_tokens']:.0f}/{c['mean_wb_thinking_tokens']:.0f} | "
                f"{c['mean_bb_acc']:.3f} | {c['mean_wb_acc']:.3f} | "
                f"{c['delta_wb']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] | "
                f"{c['p_value']:.4g} | {c.get('p_holm_cell', c['p_value']):.4g} | {c['p_holm']:.4g} | "
                f"{c['gamma_wb_bits']:+.2f} | {c['decoy_trap_rate']:.2f} |"
            )
        lines.append("")

    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage report: H1-H4 verdicts from one JSONL log.")
    parser.add_argument("jsonl", help="Path to a JSONL trial log.")
    parser.add_argument("--json-out", default="", help="Optional path to save full JSON summary.")
    parser.add_argument("--md-out", default="", help="Optional path to save Markdown report.")
    args = parser.parse_args(argv)

    records = load_jsonl(Path(args.jsonl))
    total_cost = sum(float(r.get("cost_usd", 0.0)) for r in records)
    res1 = analyze_exp1(records)
    res2 = analyze_exp2(records)
    md = format_markdown_report(res1, res2, total_cost)
    print(md)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"total_cost_usd": total_cost, "exp1": res1, "exp2": res2}, indent=2),
            encoding="utf-8",
        )
    if args.md_out:
        Path(args.md_out).write_text(md, encoding="utf-8")
    return 0




if __name__ == "__main__":
    raise SystemExit(main())
