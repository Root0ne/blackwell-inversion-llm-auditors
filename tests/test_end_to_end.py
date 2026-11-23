"""Zero-cost end-to-end runs of the whole pipeline with the mock client."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from blackwell_inversion.analysis.legacy_report import analyze_exp1, analyze_exp2
from blackwell_inversion.cli import main as runner_main

E2E_ARGS = [
    "--dry-run",
    "--experiment",
    "all",
    "--models",
    "small-tier,strong-tier",
    "--family",
    "4xor_n40",
    "--effort",
    "medium",
    "--regimes",
    "control_impossible,vbb_hard,transparent",
    "--substrates",
    "code_clean,code_obf,relu_weights",
    "--tracks",
    "tool_python",
    "--exp1-pairs",
    "10",
    "--variants",
    "wb_clean,wb_inv_code,wb_inv_relu",
    "--budget-tracks",
    "matched_total_budget,matched_medium_budget,matched_query_free_code",
    "--exp2-seeds",
    "10",
]


def _quiet_main(argv) -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        return runner_main(argv)


class TestEndToEndDryRunPipeline(unittest.TestCase):
    def test_dry_run_exp1_and_exp2_then_analyze_and_resume(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vbb_e2e_") as td:
            out_jsonl = Path(td) / "dry_results.jsonl"
            self.assertEqual(_quiet_main(E2E_ARGS + ["--out", str(out_jsonl)]), 0)
            self.assertTrue(out_jsonl.exists())

            records = [json.loads(ln) for ln in out_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
            # Exp1: 2 models * 3 regimes * (1 BB + 3 WB) * 20 = 480
            # Exp2: 2 models * 3 tracks * (1 BB + 3 WB) * 10 = 240
            self.assertEqual(len(records), 720)
            self.assertEqual(sum(float(r["cost_usd"]) for r in records), 0.0)
            self.assertTrue(all(r.get("effort") == "medium" for r in records))

            # Resuming skips all 720 trials and appends nothing.
            self.assertEqual(_quiet_main(E2E_ARGS + ["--out", str(out_jsonl)]), 0)
            records_after = [ln for ln in out_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(records_after), 720)

            res1 = analyze_exp1(records)
            res2 = analyze_exp2(records)
            self.assertTrue(res1["leak_gate_passed"])
            self.assertEqual(res1["h1_verdict"], "SUPPORTED")
            self.assertEqual(res2["h2_verdict"], "SUPPORTED_BOTH_MECH1_AND_MECH2")
            # The wb_clean control is positive or zero while wb_inv_* are strongly negative.
            clean_deltas = [c["delta_wb"] for c in res2["comparisons"] if c["variant"] == "wb_clean"]
            inv_deltas = [c["delta_wb"] for c in res2["comparisons"] if c["variant"].startswith("wb_inv_")]
            self.assertTrue(all(d >= -0.02 for d in clean_deltas))
            self.assertTrue(all(d <= -0.30 for d in inv_deltas))

    def test_dry_run_is_deterministic_apart_from_timing(self) -> None:
        argv = ["--dry-run", "--experiment", "all", "--models", "haiku", "--exp1-pairs", "1", "--regimes", "vbb_hard",
                "--substrates", "relu_weights", "--exp2-seeds", "2", "--budget-tracks", "matched_medium_budget",
                "--protocol", "harness_check_tools", "--log-tool-inputs"]
        runs = []
        with tempfile.TemporaryDirectory() as td:
            for name in ("a", "b"):
                out = Path(td) / f"{name}.jsonl"
                self.assertEqual(_quiet_main(argv + ["--out", str(out)]), 0)
                recs = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
                for r in recs:
                    r.pop("wall_s")
                runs.append(recs)
        self.assertEqual(len(runs[0]), 4 + 8)
        self.assertEqual(runs[0], runs[1])


if __name__ == "__main__":
    unittest.main()
