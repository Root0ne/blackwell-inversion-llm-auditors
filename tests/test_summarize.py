"""The summary command: same numbers as the paper's report, honest about incomplete and empty logs."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from blackwell_inversion.analysis import summarize
from blackwell_inversion.analysis.stats import load_jsonl
from blackwell_inversion.paths import C2_FILE, C3_MERGED_FILE, MAIN_REPORT_JSON


def _run(argv: List[str]) -> tuple:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = summarize.main(argv)
    return rc, out.getvalue()


class TestAgainstThePaper(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(MAIN_REPORT_JSON.read_text(encoding="utf-8"))
        cls.records = load_jsonl(C2_FILE) + load_jsonl(C3_MERGED_FILE)

    def _summary_of(self, model: str) -> Dict[str, Any]:
        return summarize.summarize_records([r for r in self.records if r["model"] == model])

    def test_exp2_numbers_equal_the_report(self) -> None:
        for model in ("claude-haiku-5-5", "gemini-3.8-flash", "gemma-4-31b-it"):
            mine = self._summary_of(model)["exp2"]
            paper = [c for c in self.report["exp2"]["cells"] if c["model"] == model]
            self.assertEqual(len(mine), len(paper))
            for cell, ref in zip(mine, paper):
                self.assertEqual(
                    (cell["effort"], cell["budget_track"], cell["task_type"], cell["max_tokens"]),
                    (ref["effort"], ref["budget_track"], ref["task_type"], ref["max_tokens"]),
                )
                self.assertEqual((cell["bb"]["n"], cell["bb"]["acc"]), (ref["bb_n"], ref["bb_acc"]))
                self.assertEqual(sorted(cell["variants"]), sorted(ref["variants"]))
                for var, v in cell["variants"].items():
                    r = ref["variants"][var]
                    self.assertEqual(
                        (v["n"], v["acc"], v["trap_rate"], v["parse_fail_rate"], v["mean_turns"]),
                        (r["n"], r["acc"], r["decoy_rate"], 1.0 - r["parse_rate"], r["mean_turns"]),
                    )
                    self.assertEqual(
                        (v["delta_wb"], v["ci_lo"], v["ci_hi"], v["p_value"], v["n_pairs"]),
                        tuple(r["vs_bb"][k] for k in ("delta_wb", "ci_lo", "ci_hi", "p_value", "n_pairs")),
                    )
        main = [c for c in self._summary_of("claude-haiku-5-5")["exp2"]
                if (c["effort"], c["budget_track"], c["task_type"], c["max_tokens"]) == ("medium", "matched_medium_budget", "majority3", "4096")]
        self.assertTrue(main[0]["complete"])
        self.assertEqual(main[0]["seeds"], list(range(5000, 5025)))

    def test_exp1_numbers_equal_the_report(self) -> None:
        for model in ("claude-opus-5-5", "claude-sonnet-5-5"):
            mine = [r for r in self._summary_of(model)["exp1"] if r["family"] == "4xor_strict"]
            paper = [r for r in self.report["exp1_4xor_strict"] if r["model"] == model]
            self.assertEqual(len(mine), len(paper))
            for row, ref in zip(mine, paper):
                for key in ("track", "regime", "substrate", "n", "tpr", "fpr", "adv", "ci_lo", "ci_hi", "p_value"):
                    self.assertEqual(row[key], ref[key], (model, key))
                self.assertEqual(row["p_holm_model"], ref.get("p_holm_model"))

    def test_compare_paper_prints_the_paper_models(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "haiku.jsonl"
            rows = [r for r in self.records if r["run_key"].startswith("exp2|claude-haiku-5-5|tm=adaptive|ef=medium|matched_medium_budget|")]
            log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            rc, text = _run([str(log), "--compare-paper", "--preset", "paper-exp2-main"])
        self.assertEqual(rc, 0)
        self.assertIn("Preset coverage: 150 of 150 scheduled trials (complete).", text)
        self.assertIn("Paper, same condition", text)
        for model in ("claude-sonnet-5-5", "gemini-4-carbon", "gemma-4-26b-a4b-it"):
            self.assertIn(model, text)
        self.assertNotIn("INCOMPLETE", text)


class TestIncompleteAndEmptyLogs(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        main = load_jsonl(C2_FILE) + load_jsonl(C3_MERGED_FILE)
        self.rows = [r for r in main if r["run_key"].startswith("exp2|claude-haiku-5-5|tm=adaptive|ef=medium|matched_medium_budget|")
                     and r["seed"] < 5004]

    def _write(self, name: str, rows: List[Dict[str, Any]], extra: str = "") -> str:
        path = self.dir / name
        path.write_text("".join(json.dumps(r) + "\n" for r in rows) + extra, encoding="utf-8")
        return str(path)

    def test_missing_seeds_withhold_the_estimate(self) -> None:
        rows = [r for r in self.rows if not (r["variant"] == "wb_inv_code" and r["mode"] == "WB" and r["seed"] == 5002)]
        log = self._write("partial.jsonl", rows)
        rc, text = _run([log])
        self.assertEqual(rc, 0)
        self.assertIn("INCOMPLETE: wb_inv_code misses seeds [5002]", text)
        self.assertIn("Delta_WB is withheld", text)
        self.assertNotRegex(text, r"wb_inv_code \(hard-guard code\)\s+3\s+\S+\s+[+-]\d")
        rc, text = _run([log, "--allow-incomplete"])
        self.assertIn("*3", text)  # the estimate over three paired seeds, marked
        self.assertIn("computed over the paired seeds only", text)

    def test_missing_black_box_mode(self) -> None:
        summary = summarize.summarize_records([r for r in self.rows if r["mode"] == "WB"])
        self.assertFalse(summary["exp2"][0]["complete"])
        self.assertIn("no black-box trials", summary["exp2"][0]["problems"])

    def test_complete_subset_is_complete(self) -> None:
        summary = summarize.summarize_records(self.rows)
        self.assertTrue(summary["exp2"][0]["complete"])
        self.assertEqual(summary["exp2"][0]["seeds"], [5000, 5001, 5002, 5003])

    def test_exp1_cell_missing_a_pair_member(self) -> None:
        exp1 = [r for r in load_jsonl(C2_FILE) + load_jsonl(C3_MERGED_FILE) if r["run_key"].startswith("exp1|claude-opus-5-5|")
                and r["regime"] == "vbb_hard" and r["seed"] < 1300]
        exp1 = [r for r in exp1 if not (r["substrate"] == "code_clean" and r["seed"] == 1101)]
        summary = summarize.summarize_records(exp1)
        by_sub = {r["substrate"]: r for r in summary["exp1"]}
        self.assertFalse(by_sub["code_clean"]["complete"])
        self.assertEqual(by_sub["code_clean"]["problems"], ["planted trials missing for pairs [1100]"])
        self.assertTrue(by_sub["blackbox"]["complete"])
        rc, text = _run([self._write("e1.jsonl", exp1)])
        self.assertIn("INCOMPLETE: planted trials missing for pairs [1100]", text)

    def test_preset_coverage_lists_missing_trials(self) -> None:
        cov = summarize.preset_coverage(self.rows, "paper-exp2-main")["claude-haiku-5-5"]
        self.assertEqual((cov["expected"], cov["present"], len(cov["missing"])), (150, 24, 126))

    def test_unreadable_lines_and_repeated_keys_are_counted(self) -> None:
        log = self._write("messy.jsonl", self.rows + self.rows[:2] + [{"note": "not a trial"}], extra='{"run_key": "trunc')
        rc, text = _run([log])
        self.assertEqual(rc, 0)
        self.assertIn("Records: 24 usable trials; skipped 1 unreadable lines, 1 non-trial records, 2 repeated run keys.", text)

    def test_log_with_zero_usable_trials(self) -> None:
        for content in ("", '{"broken\n', '{"note": 1}\n'):
            path = self.dir / "empty.jsonl"
            path.write_text(content, encoding="utf-8")
            rc, text = _run([str(path)])
            self.assertEqual(rc, 1)
            self.assertIn("No usable trials", text)

    def test_json_output(self) -> None:
        out = self.dir / "summary.json"
        rc, _ = _run([self._write("ok.jsonl", self.rows), "--out-json", str(out)])
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["summary"]["models"]["claude-haiku-5-5"]["exp2"], 24)
        self.assertIsNone(payload["preset_coverage"])


if __name__ == "__main__":
    unittest.main()
