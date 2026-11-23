"""Result summary of your own trial log, optionally next to the paper's models.

    python -m blackwell_inversion.analysis.summarize my.jsonl [more.jsonl ...] \
        [--compare-paper] [--preset paper-exp2-main] [--allow-incomplete] [--out-json S.json]

Per model and condition it prints, for Exp 2, the black-box and white-box held-out
accuracy per variant, Delta_WB = WB - BB with its paired bootstrap 95% CI and the
one-sided sign-flip p-value, the trap rate (answer equals the decoy rule), the
parse-failure rate, and the mean turns, oracle queries, tokens and the cost; for Exp 1
the distinguishing advantage Adv = TPR - FPR with its Newcombe-Wilson CI and the exact
Fisher p-value. All statistics come from `main_report` (`exp2_paired_summary`,
`exp1_summary`), the code behind the paper's report, with the same seeds.

Completeness. Delta_WB pairs trials by seed: an Exp 2 condition is incomplete when a
mode or variant lacks seeds that other modes or variants of the same condition have, or
when it has no black-box or no white-box trials. Adv is estimated per cell from planted
and clean trials: an Exp 1 cell is incomplete when one member of a planted/clean pair is
missing (a cell with fewer pairs than its neighbours is a smaller, valid sample). For an incomplete condition the estimate is withheld and the missing seeds are
listed; `--allow-incomplete` prints the estimate over the trials that do pair, marked as
such. `--preset NAME` also checks each model's log against the full schedule of that
preset. Unreadable lines (an interrupted write) are counted and skipped; a repeated
run_key keeps its first record.

`--compare-paper` prints the paper's models in the same condition from
`reports/report_c3_unified.json`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..paths import MAIN_REPORT_JSON
from .main_report import exp1_summary, exp2_cells, exp2_paired_summary

EXPERIMENTS = ("exp1_vbb", "exp2_blackwell")
VARIANT_NAMES = {
    "wb_clean": "clean code",
    "wb_decoy_trivial_guard": "short control",
    "wb_decoy_long_refutable_guard": "refutable control",
    "wb_inv_code": "hard-guard code",
    "wb_inv_relu": "opaque weights",
}
SUBSTRATE_NAMES = {
    "blackbox": "black box",
    "code_clean": "clean code",
    "code_obf": "obfuscated code",
    "relu_weights": "sparse weights",
    "relu_weights_dense": "dense weights",
}


# ------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------
def load_log(paths: Sequence[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Usable trial records of one or more logs, and counts of what was skipped."""
    records: List[Dict[str, Any]] = []
    seen: set[str] = set()
    counts = {"lines": 0, "unreadable": 0, "not_trials": 0, "duplicates": 0}
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                counts["lines"] += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    counts["unreadable"] += 1
                    continue
                if not isinstance(rec, dict) or rec.get("experiment") not in EXPERIMENTS or "run_key" not in rec:
                    counts["not_trials"] += 1
                    continue
                if rec["run_key"] in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(rec["run_key"])
                records.append(rec)
    return records, counts


# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
def _usage(rs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean turns, queries and tokens per trial, and the summed cost."""
    if not rs:
        return {"mean_turns": None, "mean_queries": None, "mean_input_tokens": None, "mean_output_tokens": None, "cost_usd": 0.0}
    return {
        "mean_turns": float(np.mean([r.get("turns", 0) for r in rs])),
        "mean_queries": float(np.mean([r.get("queries_used", 0) for r in rs])),
        "mean_input_tokens": float(np.mean([r.get("input_tokens", 0) for r in rs])),
        "mean_output_tokens": float(np.mean([r.get("output_tokens", 0) for r in rs])),
        "cost_usd": float(sum(r.get("cost_usd", 0.0) for r in rs)),
    }


def _exp2_summaries(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stats = exp2_paired_summary(records)["cells"]
    out: List[Dict[str, Any]] = []
    for (cell, bb_map, variants_map), st in zip(exp2_cells(records), stats):
        model, effort, track, task, mtag = cell
        assert (st["model"], st["effort"], st["budget_track"], st["task_type"], st["max_tokens"]) == cell
        wb = {var: smap for (mode, var), smap in variants_map.items() if mode == "WB"}
        all_seeds = set(bb_map)
        for smap in wb.values():
            all_seeds |= set(smap)
        problems: List[str] = []
        if not bb_map:
            problems.append("no black-box trials")
        elif set(bb_map) != all_seeds:
            problems.append(f"black box misses seeds {sorted(all_seeds - set(bb_map))}")
        if not wb:
            problems.append("no white-box trials")
        variants: Dict[str, Any] = {}
        for var in sorted(wb):
            vst = st["variants"][var]
            missing = sorted(all_seeds - set(wb[var]))
            if missing:
                problems.append(f"{var} misses seeds {missing}")
            rs = list(wb[var].values())
            variants[var] = {
                "n": vst["n"],
                "acc": vst["acc"],
                "delta_wb": vst["vs_bb"]["delta_wb"],
                "ci_lo": vst["vs_bb"]["ci_lo"],
                "ci_hi": vst["vs_bb"]["ci_hi"],
                "p_value": vst["vs_bb"]["p_value"],
                "n_pairs": vst["vs_bb"]["n_pairs"],
                "trap_rate": vst["decoy_rate"],
                "parse_fail_rate": 1.0 - vst["parse_rate"],
                "exact_rule_rate": float(np.mean([1.0 if r.get("exact_rule") else 0.0 for r in rs])),
                "missing_seeds": missing,
                **_usage(rs),
            }
        bb_rs = list(bb_map.values())
        out.append(
            {
                "model": model,
                "effort": effort,
                "budget_track": track,
                "task_type": task,
                "max_tokens": mtag,
                "protocols": sorted({r.get("protocol", "standard") for r in bb_rs + [r for s in wb.values() for r in s.values()]}),
                "seeds": sorted(all_seeds),
                "complete": not problems,
                "problems": problems,
                "bb": {
                    "n": st["bb_n"],
                    "acc": st["bb_acc"],
                    "parse_fail_rate": (1.0 - st["bb_parse"]) if st["bb_parse"] is not None else None,
                    "exact_rule_rate": float(np.mean([1.0 if r.get("exact_rule") else 0.0 for r in bb_rs])) if bb_rs else None,
                    **_usage(bb_rs),
                },
                "variants": variants,
            }
        )
    return out


def _exp1_summaries(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Exp 1 cells per (family, effort); `exp1_summary` pools efforts, so it is called once per effort."""
    exp1 = [r for r in records if r.get("run_key", "").startswith("exp1")]
    out: List[Dict[str, Any]] = []
    for family, effort in sorted({(r.get("family", ""), r.get("effort", "")) for r in exp1}):
        rows = [r for r in exp1 if (r.get("family"), r.get("effort")) == (family, effort)]
        # Pairs per (model, track, regime) and substrate; seed - planted identifies a planted/clean pair.
        pairs: Dict[Tuple[str, str, str], Dict[str, Dict[int, set]]] = defaultdict(lambda: defaultdict(lambda: {0: set(), 1: set()}))
        trials: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            planted = int(r.get("planted", 0))
            pairs[(r["model"], r["track"], r["regime"])][r["substrate"]][planted].add(int(r["seed"]) - planted)
            trials[(r["model"], r["track"], r["regime"], r["substrate"])].append(r)
        for st in exp1_summary(rows, family):
            # Adv is estimated per cell, so a cell needs both members of each of its pairs;
            # a cell with fewer pairs than its neighbours is still a valid (smaller) estimate.
            group = pairs[(st["model"], st["track"], st["regime"])]
            mine = group[st["substrate"]]
            problems = []
            if mine[1] - mine[0]:
                problems.append(f"clean trials missing for pairs {sorted(mine[1] - mine[0])}")
            if mine[0] - mine[1]:
                problems.append(f"planted trials missing for pairs {sorted(mine[0] - mine[1])}")
            notes = [] if "blackbox" in group else ["no black-box cell in this regime to compare with"]
            out.append(
                {
                    **{k: st[k] for k in ("family", "model", "track", "regime", "substrate", "n", "n_pos", "n_neg", "tpr", "fpr", "adv", "ci_lo", "ci_hi", "p_value")},
                    "effort": effort,
                    "p_holm_model": st.get("p_holm_model"),
                    "parse_fail_rate": 1.0 - st["parse_rate"],
                    "complete": not problems,
                    "problems": problems,
                    "notes": notes,
                    **_usage(trials[(st["model"], st["track"], st["regime"], st["substrate"])]),
                }
            )
    return out


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The summary of usable trial records (see the module docstring)."""
    models: Dict[str, Any] = {}
    for model in sorted({r["model"] for r in records}):
        rs = [r for r in records if r["model"] == model]
        models[model] = {
            "trials": len(rs),
            "exp1": sum(1 for r in rs if r["experiment"] == "exp1_vbb"),
            "exp2": sum(1 for r in rs if r["experiment"] == "exp2_blackwell"),
            "input_tokens": int(sum(r.get("input_tokens", 0) for r in rs)),
            "output_tokens": int(sum(r.get("output_tokens", 0) for r in rs)),
            "thinking_tokens": int(sum(r.get("thinking_tokens", 0) for r in rs)),
            "cost_usd": float(sum(r.get("cost_usd", 0.0) for r in rs)),
            "mean_turns": float(np.mean([r.get("turns", 0) for r in rs])),
        }
    return {"models": models, "exp2": _exp2_summaries(records), "exp1": _exp1_summaries(records)}


def preset_coverage(records: Sequence[Dict[str, Any]], preset: str) -> Dict[str, Dict[str, Any]]:
    """Per model: how many of the preset's scheduled trials the log holds, and which are missing."""
    from ..cli import build_jobs, build_parser
    from ..presets import apply_preset

    out: Dict[str, Dict[str, Any]] = {}
    for model in sorted({r["model"] for r in records}):
        args = build_parser().parse_args(["--preset", preset, "--models", model])
        apply_preset(args, {})
        expected = [job["run_key"] for _kind, job in build_jobs(args, [model])]
        present = {r["run_key"] for r in records if r["model"] == model}
        missing = [k for k in expected if k not in present]
        out[model] = {"expected": len(expected), "present": len(expected) - len(missing), "missing": missing}
    return out


# ------------------------------------------------------------------------------
# Text output
# ------------------------------------------------------------------------------
def _f(val: Optional[float], fmt: str) -> str:
    return "-" if val is None else format(val, fmt)


def _paper_exp2_rows(paper: Dict[str, Any], cell: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    keys = ("budget_track", "task_type", "max_tokens")
    same = [c for c in paper["exp2"]["cells"] if all(c[k] == cell[k] for k in keys)]
    rows = [c for c in same if c["effort"] == cell["effort"]]
    note = ""
    if not rows and same:
        rows = [c for c in same if c["effort"] == "medium"]
        note = f" (paper rows at effort=medium; this log has effort={cell['effort']})"
    return rows, note


def _print_exp2(cell: Dict[str, Any], allow_incomplete: bool, paper: Optional[Dict[str, Any]]) -> None:
    protocols = ",".join(cell["protocols"])
    print(
        f"\n  Exp 2 | effort={cell['effort']} | {cell['budget_track']} | {cell['task_type']} | "
        f"token cap {cell['max_tokens']} | protocol {protocols} | seeds {len(cell['seeds'])}"
    )
    show_delta = cell["complete"] or allow_incomplete
    if not cell["complete"]:
        print("    INCOMPLETE: " + "; ".join(cell["problems"]))
        print("    Delta_WB is " + ("computed over the paired seeds only." if allow_incomplete else "withheld (use --allow-incomplete to see it over the paired seeds)."))
    print(
        f"    {'condition':48s} {'N':>3s} {'acc':>6s} {'Delta_WB [95% CI]':>24s} {'p':>7s} {'trap':>6s} "
        f"{'parse-fail':>10s} {'turns':>5s} {'queries':>7s} {'tokens in/out per trial':>24s} {'cost $':>9s}"
    )
    bb = cell["bb"]
    if bb["n"]:
        print(
            f"    {'black box':48s} {bb['n']:3d} {_f(bb['acc'], '6.3f')} {'':>24s} {'':>7s} {'':>6s} "
            f"{_f(bb['parse_fail_rate'], '10.1%')} {_f(bb['mean_turns'], '5.2f')} {_f(bb['mean_queries'], '7.1f')} "
            f"{_f(bb['mean_input_tokens'], '11,.0f')} / {_f(bb['mean_output_tokens'], '10,.0f')} {bb['cost_usd']:9.4f}"
        )
    for var, v in cell["variants"].items():
        name = f"{var} ({VARIANT_NAMES.get(var, var)})"
        if show_delta and v["n_pairs"]:
            delta = f"{v['delta_wb']:+.3f} [{v['ci_lo']:+.3f},{v['ci_hi']:+.3f}]"
            if not cell["complete"]:
                delta += f"*{v['n_pairs']}"
            p = f"{v['p_value']:.4f}"
        else:
            delta, p = "-", "-"
        print(
            f"    {name:48s} {v['n']:3d} {v['acc']:6.3f} {delta:>24s} {p:>7s} {v['trap_rate']:6.1%} "
            f"{v['parse_fail_rate']:10.1%} {_f(v['mean_turns'], '5.2f')} {_f(v['mean_queries'], '7.1f')} "
            f"{_f(v['mean_input_tokens'], '11,.0f')} / {_f(v['mean_output_tokens'], '10,.0f')} {v['cost_usd']:9.4f}"
        )
    if not cell["complete"] and allow_incomplete:
        print("    *N: estimate over N paired seeds of an incomplete condition.")
    if paper is not None:
        rows, note = _paper_exp2_rows(paper, cell)
        if not rows:
            print("    Paper: no model ran this condition.")
            return
        variants = [v for v in VARIANT_NAMES if v in cell["variants"]] or list(VARIANT_NAMES)
        print(f"    Paper, same condition{note}: BB accuracy, then Delta_WB per variant (trap rate on hard-guard code)")
        print(f"      {'model':24s} {'N':>3s} {'BB':>6s} " + " ".join(f"{VARIANT_NAMES[v]:>17s}" for v in variants) + f" {'trap':>6s}")
        for c in rows:
            vals = []
            for v in variants:
                vd = c["variants"].get(v)
                vals.append(f"{vd['vs_bb']['delta_wb']:+17.3f}" if vd and vd["vs_bb"]["n_pairs"] else f"{'-':>17s}")
            trap = c["variants"].get("wb_inv_code", {}).get("decoy_rate")
            print(f"      {c['model']:24s} {c['bb_n']:3d} {_f(c['bb_acc'], '6.3f')} " + " ".join(vals) + f" {_f(trap, '6.1%')}")


def _print_exp1(rows: List[Dict[str, Any]], allow_incomplete: bool, paper: Optional[Dict[str, Any]]) -> None:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["family"], r["effort"], r["track"])].append(r)
    for (family, effort, track), rs in sorted(groups.items()):
        print(f"\n  Exp 1 | {family} | effort={effort} | {track}")
        print(
            f"    {'regime':19s} {'access':16s} {'N':>3s} {'TPR':>5s} {'FPR':>5s} {'Adv [95% CI]':>24s} {'p':>9s} "
            f"{'parse-fail':>10s} {'turns':>5s} {'queries':>7s} {'cost $':>9s}"
        )
        for r in rs:
            if r["complete"] or allow_incomplete:
                adv = f"{r['adv']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]" + ("" if r["complete"] else "*")
                p = f"{r['p_value']:.2e}"
            else:
                adv, p = "-", "-"
            print(
                f"    {r['regime']:19s} {SUBSTRATE_NAMES.get(r['substrate'], r['substrate']):16s} {r['n']:3d} "
                f"{r['tpr']:5.2f} {r['fpr']:5.2f} {adv:>24s} {p:>9s} {r['parse_fail_rate']:10.1%} "
                f"{_f(r['mean_turns'], '5.2f')} {_f(r['mean_queries'], '7.1f')} {r['cost_usd']:9.4f}"
            )
            if not r["complete"]:
                print("      INCOMPLETE: " + "; ".join(r["problems"]))
            for note in r["notes"]:
                print(f"      note: {note}")
        if paper is not None:
            key = f"exp1_{family}"
            prows = [
                p
                for p in paper.get(key, [])
                if p["track"] == track and any((p["regime"], p["substrate"]) == (r["regime"], r["substrate"]) for r in rs)
            ]
            if not prows:
                print("    Paper: no model ran these cells.")
                continue
            print("    Paper, same cells: Adv [95% CI] (N)")
            for p in prows:
                print(
                    f"      {p['model']:24s} {p['regime']:19s} {SUBSTRATE_NAMES.get(p['substrate'], p['substrate']):16s} "
                    f"{p['adv']:+.3f} [{p['ci_lo']:+.3f},{p['ci_hi']:+.3f}] ({p['n']})"
                )


def print_summary(
    summary: Dict[str, Any],
    counts: Dict[str, int],
    *,
    allow_incomplete: bool = False,
    paper: Optional[Dict[str, Any]] = None,
    coverage: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    usable = sum(m["trials"] for m in summary["models"].values())
    print(
        f"Records: {usable} usable trials; skipped {counts['unreadable']} unreadable lines, "
        f"{counts['not_trials']} non-trial records, {counts['duplicates']} repeated run keys."
    )
    for model, m in summary["models"].items():
        print(f"\n=== {model}: {m['trials']} trials (Exp 1: {m['exp1']}, Exp 2: {m['exp2']}) ===")
        print(
            f"  Tokens: {m['input_tokens']:,} input, {m['output_tokens']:,} output ({m['thinking_tokens']:,} reasoning); "
            f"{m['mean_turns']:.2f} model turns (requests) per trial; estimated cost ${m['cost_usd']:.4f}"
        )
        if coverage is not None and model in coverage:
            cov = coverage[model]
            state = "complete" if not cov["missing"] else f"{len(cov['missing'])} trials missing"
            print(f"  Preset coverage: {cov['present']} of {cov['expected']} scheduled trials ({state}).")
            for key in cov["missing"][:10]:
                print(f"    missing: {key}")
            if len(cov["missing"]) > 10:
                print(f"    ... and {len(cov['missing']) - 10} more")
        for cell in summary["exp2"]:
            if cell["model"] == model:
                _print_exp2(cell, allow_incomplete, paper)
        exp1_rows = [r for r in summary["exp1"] if r["model"] == model]
        if exp1_rows:
            _print_exp1(exp1_rows, allow_incomplete, paper)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from ..presets import PRESETS

    ap = argparse.ArgumentParser(description="Summarise your own trial log(s) and optionally compare with the paper.")
    ap.add_argument("logs", nargs="+", help="JSONL log(s) written by python -m blackwell_inversion.")
    ap.add_argument("--compare-paper", action="store_true", help="Print the paper's models in the same conditions.")
    ap.add_argument("--paper-report", default=str(MAIN_REPORT_JSON), help="The paper's main report JSON.")
    ap.add_argument("--preset", choices=tuple(PRESETS), default=None, help="Also check each model's log against this preset's schedule.")
    ap.add_argument("--allow-incomplete", action="store_true", help="Show estimates of incomplete conditions over the trials that pair.")
    ap.add_argument("--out-json", default="", help="Also write the summary as JSON.")
    args = ap.parse_args(argv)

    records, counts = load_log([Path(p) for p in args.logs])
    if not records:
        print(
            f"No usable trials in {', '.join(args.logs)} ({counts['lines']} lines: {counts['unreadable']} unreadable, "
            f"{counts['not_trials']} not trial records). Nothing to summarise."
        )
        return 1
    summary = summarize_records(records)
    coverage = preset_coverage(records, args.preset) if args.preset else None
    paper = json.loads(Path(args.paper_report).read_text(encoding="utf-8")) if args.compare_paper else None
    print_summary(summary, counts, allow_incomplete=args.allow_incomplete, paper=paper, coverage=coverage)
    if args.out_json:
        payload = {"counts": counts, "summary": summary, "preset_coverage": coverage}
        Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
