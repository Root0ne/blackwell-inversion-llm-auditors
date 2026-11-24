"""The summary command: same numbers as the paper's report, honest about incomplete and empty logs."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from blackwell_inversion import cli
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

    def test_printed_and_json_numbers_are_the_report_values(self) -> None:
        # Literal values of reports/report_c3_unified.json (Haiku 5.5, main Exp 2 condition).
        literal = {
            "bb": (25, 0.9716),
            "wb_inv_code": (25, 0.8668, 0.28, -0.1048, -0.1868, -0.0344, 0.0173982601739826),
            "wb_clean": (25, 1.0, 0.0, 0.028399999999999998, 0.0, 0.0768, 0.24477552244775522),
        }
        ref = next(c for c in self.report["exp2"]["cells"] if c["model"] == "claude-haiku-5-5" and
                   (c["effort"], c["budget_track"], c["task_type"], c["max_tokens"]) == ("medium", "matched_medium_budget", "majority3", "4096"))
        self.assertEqual((ref["bb_n"], ref["bb_acc"]), literal["bb"])
        for var in ("wb_inv_code", "wb_clean"):
            r = ref["variants"][var]
            self.assertEqual((r["n"], r["acc"], r["decoy_rate"], *(r["vs_bb"][k] for k in ("delta_wb", "ci_lo", "ci_hi", "p_value"))),
                             literal[var])
        with tempfile.TemporaryDirectory() as td:
            log, out = Path(td) / "haiku.jsonl", Path(td) / "s.json"
            rows = [r for r in self.records if r["run_key"].startswith("exp2|claude-haiku-5-5|tm=adaptive|ef=medium|matched_medium_budget|")]
            log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            rc, text = _run([str(log), "--out-json", str(out)])
            cell = json.loads(out.read_text(encoding="utf-8"))["summary"]["exp2"][0]
        self.assertEqual(rc, 0)
        self.assertEqual((cell["bb"]["n"], cell["bb"]["acc"]), literal["bb"])
        for var in ("wb_inv_code", "wb_clean"):
            v = cell["variants"][var]
            self.assertEqual((v["n"], v["acc"], v["trap_rate"], v["delta_wb"], v["ci_lo"], v["ci_hi"], v["p_value"]), literal[var])
        self.assertRegex(text, r"black box\s+25\s+0\.972\s")
        self.assertRegex(text, r"wb_inv_code \(hard-guard code\)\s+25\s+0\.867\s+-0\.105 \[-0\.187,-0\.034\]\s+0\.0174\s+28\.0%")
        self.assertRegex(text, r"wb_clean \(clean code\)\s+25\s+1\.000\s+\+0\.028 \[\+0\.000,\+0\.077\]\s+0\.2448\s+0\.0%")

    def test_paper_thinking_mode_per_effort(self) -> None:
        # The comparison notes rely on this mapping; it holds for every record behind the main report.
        for rec in self.records:
            self.assertEqual(rec["thinking_mode"], summarize.paper_thinking_mode(rec["effort"]), rec["run_key"])

    def test_model_counts_cited_in_the_documentation(self) -> None:
        # README and docs/evaluating_your_model.md: 13 models in the main Exp 2 condition, 5 in the Exp 1 design.
        main = {c["model"] for c in self.report["exp2"]["cells"]
                if (c["effort"], c["budget_track"], c["task_type"], c["max_tokens"]) == ("medium", "matched_medium_budget", "majority3", "4096")}
        self.assertEqual(len(main), 13)
        self.assertEqual(len({r["model"] for r in self.report["exp1_4xor_strict"]}), 5)
        self.assertEqual(len({c["model"] for c in self.report["exp2"]["cells"]}), 15)

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

    def test_black_box_missing_a_seed(self) -> None:
        rows = [r for r in self.rows if not (r["mode"] == "BB" and r["seed"] == 5002)]
        summary = summarize.summarize_records(rows)
        self.assertFalse(summary["exp2"][0]["complete"])
        self.assertEqual(summary["exp2"][0]["problems"], ["black box misses seeds [5002]"])
        rc, text = _run([self._write("nobb.jsonl", rows)])
        self.assertIn("INCOMPLETE: black box misses seeds [5002]", text)

    def test_cost_and_usage_per_condition(self) -> None:
        summary = summarize.summarize_records(self.rows)
        cell = summary["exp2"][0]
        for key, rs in [("bb", [r for r in self.rows if r["mode"] == "BB"])] + [
                (var, [r for r in self.rows if r["mode"] == "WB" and r["variant"] == var]) for var in cell["variants"]]:
            got = cell["bb"] if key == "bb" else cell["variants"][key]
            self.assertEqual(len(rs), 4)
            self.assertAlmostEqual(got["cost_usd"], sum(r["cost_usd"] for r in rs))
            self.assertAlmostEqual(got["mean_queries"], sum(r["queries_used"] for r in rs) / 4)
            self.assertAlmostEqual(got["mean_turns"], sum(r["turns"] for r in rs) / 4)
            self.assertAlmostEqual(got["mean_output_tokens"], sum(r["output_tokens"] for r in rs) / 4)
            self.assertGreater(got["cost_usd"], 0)
        self.assertNotEqual(cell["variants"]["wb_inv_code"]["mean_queries"], cell["bb"]["mean_queries"])

    def test_one_seed_gives_a_point_value_only(self) -> None:
        rows = [r for r in self.rows if r["seed"] == 5000]
        out = self.dir / "one.json"
        rc, text = _run([self._write("one.jsonl", rows), "--out-json", str(out)])
        v = json.loads(out.read_text(encoding="utf-8"))["summary"]["exp2"][0]["variants"]["wb_inv_code"]
        self.assertEqual((v["n_pairs"], v["ci_lo"], v["ci_hi"], v["p_value"]), (1, None, None, None))
        line = next(line for line in text.splitlines() if "wb_inv_code (hard-guard code)" in line)
        self.assertRegex(line, r"\s[+-]\d\.\d{3}\s+-\s")  # the point value, no interval, no p-value
        self.assertNotIn("[", line)
        self.assertIn("fewer than 2 paired seeds: no interval or p-value can be estimated", text)

    def test_a_second_record_of_the_same_trial_is_skipped(self) -> None:
        # A token cap of 4096 is the default: "|mt=4096" names the same condition as no tag.
        original = next(r for r in self.rows if r["mode"] == "WB" and r["variant"] == "wb_inv_code" and r["seed"] == 5001)
        twin = dict(original, heldout_acc=1.0 - original["heldout_acc"],
                    run_key=original["run_key"].replace("|matched_medium_budget|", "|mt=4096|matched_medium_budget|"))
        for order, kept in (([original, twin], original), ([twin, original], twin)):
            rows = [r for r in self.rows if r is not original] + order
            records, counts = summarize.load_log([Path(self._write("twin.jsonl", rows))])
            self.assertEqual((len(records), counts["duplicates"]), (24, 1))
            acc = summarize.summarize_records(records)["exp2"][0]["variants"]["wb_inv_code"]["acc"]
            accs = [r["heldout_acc"] for r in self.rows if r["variant"] == "wb_inv_code" and r["mode"] == "WB" and r is not original]
            self.assertAlmostEqual(acc, (sum(accs) + kept["heldout_acc"]) / 4)

    def test_effort_without_paper_rows_compares_with_medium(self) -> None:
        rows = [dict(r, effort="xhigh", run_key=r["run_key"].replace("|ef=medium|", "|ef=xhigh|")) for r in self.rows]
        rc, text = _run([self._write("xhigh.jsonl", rows), "--compare-paper"])
        self.assertIn("Paper, same condition (paper rows at effort=medium; this log has effort=xhigh)", text)
        self.assertIn("claude-sonnet-5-5", text)

    def test_missing_paper_report_is_a_clear_error(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as ctx:
            summarize.main([self._write("ok.jsonl", self.rows), "--compare-paper", "--paper-report", str(self.dir / "none.json")])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("--paper-report", err.getvalue())
        self.assertIn("pip install -e .", err.getvalue())

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
        [cov] = summarize.preset_coverage(self.rows, "paper-exp2-main")["claude-haiku-5-5"]
        self.assertEqual((cov["expected"], cov["present"], len(cov["missing"])), (150, 24, 126))
        self.assertEqual((cov["thinking_mode"], cov["effort"]), ("adaptive", "medium"))

    def test_unreadable_lines_and_repeated_keys_are_counted(self) -> None:
        log = self._write("messy.jsonl", self.rows + self.rows[:2] + [{"note": "not a trial"}], extra='{"run_key": "trunc')
        rc, text = _run([log])
        self.assertEqual(rc, 0)
        self.assertIn("Records: 24 usable trials; skipped 1 unreadable lines, 1 non-trial records, 2 repeated trials.", text)

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


class TestConditionsOfOneLog(unittest.TestCase):
    """A log holding several runs of one model: smoke, --effort "", --thinking-mode disabled, verify_first."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.log = Path(cls.tmp.name) / "mixed.jsonl"
        exp2 = ["--experiment", "exp2", "--variants", "wb_clean,wb_inv_code", "--budget-tracks", "matched_medium_budget",
                "--exp2-seeds", "1"]
        runs = [["--preset", "smoke"], ["--preset", "smoke", "--effort", ""], ["--preset", "smoke", "--thinking-mode", "disabled"],
                exp2 + ["--protocol", "verify_first"]]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for argv in runs:
                assert cli.main(argv + ["--dry-run", "--models", "x", "--out", str(cls.log)]) == 0

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_every_condition_is_its_own_cell(self) -> None:
        records, counts = summarize.load_log([self.log])
        self.assertEqual((len(records), counts["duplicates"]), (12, 0))
        cells = summarize.summarize_records(records)["exp2"]
        self.assertEqual(sorted((c["effort"], c["condition"]["thinking_mode"], c["condition"]["protocol"]) for c in cells), [
            ("default", "adaptive", "standard"), ("medium", "adaptive", "standard"),
            ("medium", "adaptive", "verify_first"), ("medium", "disabled", "standard")])
        for cell in cells:
            self.assertEqual((cell["bb"]["n"], sorted(cell["variants"])), (1, ["wb_clean", "wb_inv_code"]))
            self.assertTrue(cell["complete"])
        rc, text = _run([str(self.log), "--compare-paper"])
        self.assertIn("Records: 12 usable trials", text)
        headers = [line.strip() for line in text.splitlines() if line.strip().startswith("Exp 2 |")]
        self.assertEqual(len(headers), 4)
        self.assertIn("thinking_mode=disabled | protocol=standard", text)
        self.assertIn("thinking_mode=adaptive | protocol=verify_first", text)
        self.assertIn("Paper: the main report has the standard protocol only", text)
        self.assertIn("(this log has thinking mode disabled; the paper's rows ran with adaptive)", text)

    def test_preset_coverage_follows_the_logged_reasoning_settings(self) -> None:
        rc, text = _run([str(self.log), "--preset", "smoke"])
        self.assertIn("Preset coverage: 3 of 3 scheduled trials (complete).", text)
        self.assertIn("Preset coverage (thinking mode adaptive, effort default): 3 of 3 scheduled trials (complete).", text)
        self.assertIn("Preset coverage (thinking mode disabled, effort medium): 3 of 3 scheduled trials (complete).", text)
        self.assertNotIn("missing", text)


if __name__ == "__main__":
    unittest.main()
