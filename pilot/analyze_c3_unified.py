"""Unified C1 + C2 + C3 statistical analysis & dataset merger for the
Observer-Relative VBB & Computational Blackwell Inversion study.

Merges all C3 expansion JSONLs cleanly (deduplicated by run_key) into
`pilot/results_c3_expansion.jsonl` (when --write-merged is passed) WITHOUT
modifying the immutable `pilot/results_c1_haiku55.jsonl` or
`pilot/results_c2_pilot.jsonl` logs.

Computes exact Fisher, Newcombe-Wilson 95% CI, paired sign-flip permutation,
paired bootstrap 95% CI, Holm-Bonferroni FWER adjustments, 3-way loss
decomposition, and full telemetry across all 3 families (Anthropic Claude,
Google Gemini, Open-Weight Gemma 4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .analyze import fisher_adv_test, holm_bonferroni, paired_delta_stats

ROOT = Path(__file__).resolve().parent

C1_FILE = ROOT / "results_c1_haiku55.jsonl"
C2_FILE = ROOT / "results_c2_pilot.jsonl"
C3_SOURCES = [
    ROOT / "results_c3_anth_key1.jsonl",
    ROOT / "results_c3_anth_key2.jsonl",
    ROOT / "results_c3_gemini.jsonl",
    ROOT / "results_c3_gemini31_free.jsonl",
    ROOT / "results_c3_gemma.jsonl",
    ROOT / "results_c3_gemma31_clean_k1.jsonl",
    ROOT / "results_c3_gemma31_controls.jsonl",
    ROOT / "results_c3_gemma26.jsonl",
    ROOT / "results_c3_gemma26_k1.jsonl",
]
C3_MERGED = ROOT / "results_c3_expansion.jsonl"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(json.loads(s))
    return out


def merge_c3(write_merged: bool = False) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for src in C3_SOURCES:
        for rec in load_jsonl(src):
            rk = rec["run_key"]
            # Enforce exact balanced N for Gemma 4 open-weight models if extra seeds exist
            if rec.get("model") == "gemma-4-31b-it" and rec.get("seed", 0) >= 5015:
                continue
            if rec.get("model") == "gemma-4-26b-a4b-it" and rec.get("seed", 0) >= 5010:
                continue
            if rk not in seen:
                seen[rk] = rec
                order.append(rk)
    merged = [seen[rk] for rk in order]
    if write_merged:
        with C3_MERGED.open("w", encoding="utf-8") as f:
            for rec in merged:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return merged


def exp1_summary(rows: Sequence[Dict[str, Any]], family: str) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("run_key", "").startswith("exp1") and r.get("family") == family:
            key = (r["model"], r["track"], r["regime"], r["substrate"])
            groups[key].append(r)

    out: List[Dict[str, Any]] = []
    for (model, track, regime, substrate), rs in sorted(groups.items()):
        b1 = [r for r in rs if r.get("planted") == 1]
        b0 = [r for r in rs if r.get("planted") == 0]
        tp = sum(1 for r in b1 if r.get("decision_int") == 1)
        fp = sum(1 for r in b0 if r.get("decision_int") == 1)
        st = fisher_adv_test(tp, len(b1), fp, len(b0), alternative="greater")
        ok = sum(1 for r in rs if r.get("parse_ok"))
        tr = sum(1 for r in rs if r.get("truncated_turns", 0) > 0)
        mean_think = float(np.mean([r.get("thinking_tokens", 0) for r in rs]))
        out.append(
            {
                "family": family,
                "model": model,
                "track": track,
                "regime": regime,
                "substrate": substrate,
                "n": len(rs),
                "n_pos": len(b1),
                "n_neg": len(b0),
                "tp": tp,
                "fp": fp,
                "parse_ok": ok,
                "parse_rate": ok / len(rs) if rs else 0.0,
                "trunc_trials": tr,
                "mean_think": mean_think,
                **st,
            }
        )
    # Family-wide Holm-Bonferroni for non-control WB cells per model
    by_model = defaultdict(list)
    for idx, row in enumerate(out):
        if row["regime"] != "control_impossible" and row["substrate"] != "blackbox":
            by_model[row["model"]].append(idx)
    for m_name, idxs in by_model.items():
        pvals = [out[i]["p_value"] for i in idxs]
        adj = holm_bonferroni(pvals)
        for i, pa in zip(idxs, adj):
            out[i]["p_holm_model"] = pa
            out[i]["m_holm_model"] = len(idxs)
    return out


def exp2_paired_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    # Group by (model, effort, budget_track, task_type, max_tokens_tag)
    # Note: Opus 4.6 8192 trials have run_key containing "|mt8192"
    def mt_tag(r: Dict[str, Any]) -> str:
        if "|mt=8192" in r.get("run_key", "") or r.get("max_tokens_override") == 8192:
            return "8192"
        return "4096"

    by_cell: Dict[Tuple[str, str, str, str, str], Dict[Tuple[str, str], Dict[int, Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in rows:
        if not r.get("run_key", "").startswith("exp2"):
            continue
        cell = (r["model"], r["effort"], r["budget_track"], r["task_type"], mt_tag(r))
        by_cell[cell][(r["mode"], r["variant"])][int(r["seed"])] = r

    results: List[Dict[str, Any]] = []
    for cell in sorted(by_cell.keys()):
        model, effort, track, task, mtag = cell
        variants_map = by_cell[cell]
        # Find BB map (stored under ('BB', 'wb_inv_code') or any ('BB', *))
        bb_map: Dict[int, Dict[str, Any]] = {}
        for (mode, var), smap in variants_map.items():
            if mode == "BB":
                bb_map.update(smap)
        # For Opus 4.6 8192, borrow matched BB from 4096 if not separately run
        if not bb_map and mtag == "8192":
            alt = by_cell.get((model, effort, track, task, "4096"), {})
            for (mode, var), smap in alt.items():
                if mode == "BB":
                    bb_map.update(smap)
        # For matched_wide_equal_budget (8t/8t), WB rows live in matched_query_free_code (4t/8t)!
        if track == "matched_wide_equal_budget":
            alt = by_cell.get((model, effort, "matched_query_free_code", task, mtag), {})
            for (mode, var), smap in alt.items():
                if mode == "WB" and (mode, var) not in variants_map:
                    variants_map[(mode, var)] = smap

        clean_map = variants_map.get(("WB", "wb_clean"), {})

        cell_rec: Dict[str, Any] = {
            "model": model,
            "effort": effort,
            "budget_track": track,
            "task_type": task,
            "max_tokens": mtag,
            "bb_n": len(bb_map),
            "bb_acc": float(np.mean([r["heldout_acc"] for r in bb_map.values()])) if bb_map else None,
            "bb_parse": float(np.mean([1.0 if r["parse_ok"] else 0.0 for r in bb_map.values()])) if bb_map else None,
            "bb_trunc": sum(1 for r in bb_map.values() if r.get("truncated_turns", 0) > 0),
            "bb_think": float(np.mean([r.get("thinking_tokens", 0) for r in bb_map.values()])) if bb_map else None,
            "variants": {},
            "direct_diffs": {},
        }

        for (mode, var), smap in sorted(variants_map.items()):
            if mode != "WB":
                continue
            common_bb = sorted(set(smap.keys()) & set(bb_map.keys()))
            wb_vals = [smap[s]["heldout_acc"] for s in common_bb]
            bb_vals = [bb_map[s]["heldout_acc"] for s in common_bb]
            alt_dir = "greater" if var == "wb_clean" else "less"
            st_bb = paired_delta_stats(wb_vals, bb_vals, alternative=alt_dir, n_perm=10000, n_boot=10000, seed=2026)

            # Also compare vs wb_clean (crucial for Google Gemini/Gemma where BB is low!)
            st_clean = None
            if clean_map and var != "wb_clean":
                common_cl = sorted(set(smap.keys()) & set(clean_map.keys()))
                if common_cl:
                    st_clean = paired_delta_stats(
                        [smap[s]["heldout_acc"] for s in common_cl],
                        [clean_map[s]["heldout_acc"] for s in common_cl],
                        alternative="less",
                        n_perm=10000,
                        n_boot=10000,
                        seed=2026,
                    )

            rs_all = list(smap.values())
            n_all = len(rs_all)
            dec_n = sum(1 for r in rs_all if r.get("matches_decoy"))
            tr_n = sum(1 for r in rs_all if r.get("truncated_turns", 0) > 0)
            ok_n = sum(1 for r in rs_all if r.get("parse_ok"))
            # 3-way loss decomposition on all WB trials (error = 1 - heldout_acc)
            loss_decoy = sum((1.0 - r["heldout_acc"]) for r in rs_all if r.get("matches_decoy")) / n_all
            loss_trunc = sum(
                (1.0 - r["heldout_acc"]) for r in rs_all if (not r.get("matches_decoy")) and r.get("truncated_turns", 0) > 0
            ) / n_all
            loss_other = sum(
                (1.0 - r["heldout_acc"])
                for r in rs_all
                if (not r.get("matches_decoy")) and r.get("truncated_turns", 0) == 0
            ) / n_all

            cell_rec["variants"][var] = {
                "n": n_all,
                "acc": float(np.mean([r["heldout_acc"] for r in rs_all])),
                "parse_ok": ok_n,
                "parse_rate": ok_n / n_all,
                "decoy_n": dec_n,
                "decoy_rate": dec_n / n_all,
                "trunc_n": tr_n,
                "trunc_rate": tr_n / n_all,
                "mean_think": float(np.mean([r.get("thinking_tokens", 0) for r in rs_all])),
                "mean_turns": float(np.mean([r.get("turns", 0) for r in rs_all])),
                "loss_decoy": loss_decoy,
                "loss_trunc": loss_trunc,
                "loss_other": loss_other,
                "vs_bb": st_bb,
                "vs_clean": st_clean,
            }

        # Direct paired mechanism contrasts
        contrasts = [
            ("inv_code_minus_trivial", "wb_inv_code", "wb_decoy_trivial_guard", "less"),
            ("inv_code_minus_long_ref", "wb_inv_code", "wb_decoy_long_refutable_guard", "two-sided"),
            ("long_ref_minus_trivial", "wb_decoy_long_refutable_guard", "wb_decoy_trivial_guard", "less"),
            ("inv_code_minus_inv_relu", "wb_inv_code", "wb_inv_relu", "less"),
            ("long_ref_minus_inv_relu", "wb_decoy_long_refutable_guard", "wb_inv_relu", "less"),
        ]
        for cname, va, vb, alt_c in contrasts:
            ma = variants_map.get(("WB", va), {})
            mb = variants_map.get(("WB", vb), {})
            common = sorted(set(ma.keys()) & set(mb.keys()))
            if common:
                cell_rec["direct_diffs"][cname] = paired_delta_stats(
                    [ma[s]["heldout_acc"] for s in common],
                    [mb[s]["heldout_acc"] for s in common],
                    alternative=alt_c,
                    n_perm=10000,
                    n_boot=10000,
                    seed=2026,
                )

        results.append(cell_rec)
    return {"cells": results}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-merged", action="store_true")
    ap.add_argument("--out-json", default=str(ROOT / "report_c3_unified.json"))
    args = ap.parse_args()

    c1 = load_jsonl(C1_FILE)
    c2 = load_jsonl(C2_FILE)
    c3 = merge_c3(write_merged=args.write_merged)

    all_c23 = c2 + c3
    models_all = sorted({r["model"] for r in c1 + all_c23})
    total_cost_c1 = sum(r.get("cost_usd", 0.0) for r in c1)
    total_cost_c2 = sum(r.get("cost_usd", 0.0) for r in c2)
    total_cost_c3 = sum(r.get("cost_usd", 0.0) for r in c3)

    # Telemetry per model across C2 + C3
    telem: Dict[str, Dict[str, Any]] = {}
    for m in models_all:
        rs = [r for r in all_c23 if r["model"] == m]
        e1 = sum(1 for r in rs if r["run_key"].startswith("exp1"))
        e2 = sum(1 for r in rs if r["run_key"].startswith("exp2"))
        tr = sum(1 for r in rs if r.get("truncated_turns", 0) > 0)
        cost = sum(r.get("cost_usd", 0.0) for r in rs)
        telem[m] = {"exp1": e1, "exp2": e2, "total_c23": len(rs), "trunc_trials": tr, "cost_usd": cost}

    exp1_strict = exp1_summary(all_c23, "4xor_strict")
    exp1_fixed120 = exp1_summary(all_c23, "4xor_fixed120")
    exp2_res = exp2_paired_summary(all_c23)

    report = {
        "counts": {
            "c1": len(c1),
            "c2": len(c2),
            "c3": len(c3),
            "total": len(c1) + len(c2) + len(c3),
            "models_count": len(models_all),
            "models": models_all,
            "cost_c1": total_cost_c1,
            "cost_c2": total_cost_c2,
            "cost_c3": total_cost_c3,
            "cost_total": total_cost_c1 + total_cost_c2 + total_cost_c3,
        },
        "sha256": {
            "c1": sha256_file(C1_FILE),
            "c2": sha256_file(C2_FILE),
            "c3": sha256_file(C3_MERGED) if C3_MERGED.exists() else None,
        },
        "telemetry": telem,
        "exp1_4xor_strict": exp1_strict,
        "exp1_4xor_fixed120": exp1_fixed120,
        "exp2": exp2_res,
    }

    Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Concise console digest
    print(f"=== UNIFIED C1+C2+C3 DIGEST ===")
    print(f"C1={len(c1)} (${total_cost_c1:.2f}), C2={len(c2)} (${total_cost_c2:.2f}), C3={len(c3)} (${total_cost_c3:.2f}) -> TOTAL={len(c1)+len(c2)+len(c3)} trials, {len(models_all)} models")
    if C3_MERGED.exists():
        print(f"SHA-256 C3: {report['sha256']['c3']}")

    print("\n--- EXP 1: 4xor_strict (POOLED C2+C3) ---")
    for r in exp1_strict:
        if r["regime"] == "control_impossible" or r["substrate"] == "blackbox":
            continue
        print(
            f"  {r['model']:20s} {r['track']:11s} {r['regime']:11s} {r['substrate']:19s} "
            f"N={r['n']:2d} ok={r['parse_rate']:.2f} Adv={r['adv']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] "
            f"p={r['p_value']:.2e} pH={r.get('p_holm_model',1.0):.3e} (M={r.get('m_holm_model',0)})"
        )

    print("\n--- EXP 1: 4xor_fixed120 (B5 SINGLE-VAR m) ---")
    for r in exp1_fixed120:
        print(
            f"  {r['model']:20s} {r['regime']:11s} {r['substrate']:19s} "
            f"N={r['n']:2d} Adv={r['adv']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p={r['p_value']:.3e}"
        )

    print("\n--- EXP 2: KEY CELLS (LONG_REFUTABLE + GOOGLE + GEMMA + 8t/8t + OPUS46-8192 + OPUS48) ---")
    for c in exp2_res["cells"]:
        m = c["model"]
        show = (
            m.startswith(("gemini-", "gemma-"))
            or c["budget_track"] == "matched_wide_equal_budget"
            or c["max_tokens"] == "8192"
            or "wb_decoy_long_refutable_guard" in c["variants"]
            or (m == "claude-opus-4-8")
        )
        if not show:
            continue
        bb_s = f"{c['bb_acc']:.3f}" if c["bb_acc"] is not None else "N/A"
        print(f"\n  [{m} | {c['effort']} | {c['budget_track']} | {c['task_type']} | mt={c['max_tokens']}] BB={bb_s} (N={c['bb_n']})")
        for var, vd in c["variants"].items():
            d_bb = vd["vs_bb"]["delta_wb"]
            p_bb = vd["vs_bb"]["p_value"]
            cl_s = ""
            if vd["vs_clean"]:
                cl_s = f" | vsClean={vd['vs_clean']['delta_wb']:+.3f}(p={vd['vs_clean']['p_value']:.4f})"
            print(
                f"    {var:30s}: N={vd['n']:2d} ok={vd['parse_ok']:2d} tr={vd['trunc_n']:2d} "
                f"acc={vd['acc']:.3f} dec={vd['decoy_n']:2d}({100*vd['decoy_rate']:4.1f}%) "
                f"dBB={d_bb:+.3f}(p={p_bb:.4f}){cl_s}"
            )
        for cname, cd in c["direct_diffs"].items():
            print(f"      DIFF {cname:24s}: {cd['delta_wb']:+.3f} [{cd['ci_lo']:+.3f},{cd['ci_hi']:+.3f}] p={cd['p_value']:.4f} (N={cd['n_pairs']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
