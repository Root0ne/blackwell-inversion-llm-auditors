"""Trial schedules, run keys, the resumable log and the command line driver."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from blackwell_inversion import cli
from blackwell_inversion.schedule import append_jsonl, build_exp1_schedule, build_exp2_schedule, load_completed_keys

from .fakes import ScriptedClient, greedy_tool_policy

REPO_ROOT = Path(__file__).resolve().parents[1]
SMALL_EXP2 = ["--experiment", "exp2", "--models", "m1", "--variants", "wb_clean",
              "--budget-tracks", "matched_total_budget", "--exp2-seeds", "3"]


class TestSchedules(unittest.TestCase):
    def test_exp1_run_keys_and_balanced_seeds(self) -> None:
        jobs = build_exp1_schedule(["m"], "4xor_strict", ["vbb_hard"], ["code_clean"], ["tool_python"], n_pairs=2)
        self.assertEqual(len(jobs), 2 * 2 * 2)  # (BB + 1 WB) x 2 pairs x (planted, clean)
        self.assertEqual(jobs[0]["run_key"], "exp1|m|tm=adaptive|ef=medium|4xor_strict|tool_python|vbb_hard|BB|blackbox|s1000|b0")
        self.assertEqual([j["seed"] for j in jobs[:4]], [1000, 1001, 1100, 1101])
        self.assertEqual(len({j["run_key"] for j in jobs}), len(jobs))
        no_bb = build_exp1_schedule(["m"], "4xor_strict", ["vbb_hard"], ["code_clean"], ["tool_python"], 2, skip_bb=True)
        self.assertTrue(all(j["mode"] == "WB" for j in no_bb))

    def test_exp2_run_keys_tags_and_pairing(self) -> None:
        jobs = build_exp2_schedule(["m"], ["wb_clean", "wb_inv_code"], ["matched_medium_budget"], n_seeds=2)
        self.assertEqual([j["run_key"] for j in jobs[:3]], [
            "exp2|m|tm=adaptive|ef=medium|matched_medium_budget|BB|blackbox|s5000",
            "exp2|m|tm=adaptive|ef=medium|matched_medium_budget|WB|wb_clean|s5000",
            "exp2|m|tm=adaptive|ef=medium|matched_medium_budget|WB|wb_inv_code|s5000",
        ])
        self.assertEqual(jobs[0]["variant"], "wb_inv_code")
        tagged = build_exp2_schedule(["m"], ["wb_clean"], ["matched_medium_budget"], 1, effort="", task_type="dlist3",
                                     max_tokens_override=8192, protocol="harness_check")
        self.assertEqual(tagged[0]["run_key"],
                         "exp2|m|tm=adaptive|ef=default|tt=dlist3|mt=8192|pr=harness_check|matched_medium_budget|BB|blackbox|s5000")
        only_bb = build_exp2_schedule(["m"], ["wb_clean"], ["matched_medium_budget"], 2, only_bb=True)
        self.assertEqual([j["mode"] for j in only_bb], ["BB", "BB"])

    def test_completed_keys_ignore_unreadable_lines(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sub" / "log.jsonl"
            self.assertEqual(load_completed_keys(path), set())
            append_jsonl(path, {"run_key": "a"})
            append_jsonl(path, {"no_key": 1})
            with path.open("a", encoding="utf-8") as fh:
                fh.write('{"run_key": "trunc')  # an interrupted write
            self.assertEqual(load_completed_keys(path), {"a"})

    def test_a_record_after_an_interrupted_write_starts_on_a_new_line(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "log.jsonl"
            append_jsonl(path, {"run_key": "a"})
            with path.open("a", encoding="utf-8") as fh:
                fh.write('{"run_key": "trunc')  # interrupted: no closing brace, no newline
            append_jsonl(path, {"run_key": "b"})
            append_jsonl(path, {"run_key": "c"})
            self.assertEqual(path.read_text(encoding="utf-8").splitlines(),
                             ['{"run_key": "a"}', '{"run_key": "trunc', '{"run_key": "b"}', '{"run_key": "c"}'])
            self.assertEqual(load_completed_keys(path), {"a", "b", "c"})
            empty = Path(td) / "empty.jsonl"
            empty.touch()
            append_jsonl(empty, {"run_key": "x"})
            self.assertEqual(empty.read_bytes(), b'{"run_key": "x"}\n')  # no blank first line


class TestCommandLine(unittest.TestCase):
    def _main(self, argv, client) -> str:
        out = io.StringIO()
        with mock.patch.object(cli, "make_client", return_value=client), contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(out):
            self.assertEqual(cli.main(argv), 0)
        return out.getvalue()

    def test_resume_skips_logged_and_resume_from_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_path, other = Path(td) / "out.jsonl", Path(td) / "other.jsonl"
            client = ScriptedClient(greedy_tool_policy)
            self._main(SMALL_EXP2 + ["--exp2-seeds", "1", "--out", str(out_path)], client)
            self.assertEqual(len(out_path.read_text().splitlines()), 2)
            first_keys = load_completed_keys(out_path)
            append_jsonl(other, {"run_key": "exp2|m1|tm=adaptive|ef=medium|matched_total_budget|BB|blackbox|s5002"})
            log = self._main(SMALL_EXP2 + ["--out", str(out_path), "--resume-from", str(other)], client)
            self.assertIn("executed=3, skipped_resumed=3", log)
            records = [json.loads(line) for line in out_path.read_text().splitlines()]
            self.assertEqual(len(records), 5)
            self.assertEqual(len({r["run_key"] for r in records}), 5)
            self.assertTrue(first_keys <= {r["run_key"] for r in records})

    def test_spend_cap_stops_a_serial_run(self) -> None:
        # Each trial makes 3 calls at $1. Spend reaches the $4 cap with the 4th call, which opens
        # trial 2; the 5th call is refused, so trial 2 is lost and nothing after it runs.
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "out.jsonl"
            client = ScriptedClient(greedy_tool_policy, cost_per_call=1.0, max_cost_usd=4.0)
            log = self._main(SMALL_EXP2 + ["--out", str(out_path), "--max-cost-usd", "4"], client)
            self.assertIn("[STOPPED — BUDGET CAP]", log)
            self.assertIn("executed=1,", log)
            self.assertEqual(len(out_path.read_text().splitlines()), 1)
            self.assertEqual(len(client.calls), 4)

    def test_spend_cap_stops_a_threaded_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "out.jsonl"
            client = ScriptedClient(greedy_tool_policy, cost_per_call=1.0, max_cost_usd=4.0)
            log = self._main(SMALL_EXP2 + ["--out", str(out_path), "--workers", "3"], client)
            self.assertIn("[STOPPED — BUDGET CAP]", log)
            written = out_path.read_text().splitlines() if out_path.exists() else []
            self.assertLess(len(written), 6)
            # Requests already past the check when the cap is reached still complete: with 3 workers
            # and $1 calls, spend can end at most $2 above the cap.
            self.assertLessEqual(client.total_cost_usd, 6.0)

    def test_threaded_run_counts_trial_errors_and_continues(self) -> None:
        def fails_on_black_box(i: int, call):
            if "BLACK-BOX ONLY" in call["messages"][0]["content"]:
                raise RuntimeError("provider error")
            return greedy_tool_policy(i, call)

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "out.jsonl"
            log = self._main(SMALL_EXP2 + ["--exp2-seeds", "1", "--out", str(out_path), "--workers", "2"],
                             ScriptedClient(fails_on_black_box))
            self.assertIn("[TRIAL ERROR]", log)
            self.assertIn("errors=1", log)
            self.assertEqual(len(out_path.read_text().splitlines()), 1)

    def test_client_choice(self) -> None:
        args = cli.build_parser().parse_args(["--max-cost-usd", "3"])
        self.assertEqual(type(cli.make_client(args, ["gemini-3.8-flash", "gemma-4-31b-it"])).__name__, "GeminiClient")
        self.assertEqual(type(cli.make_client(args, ["gemini-3.8-flash", "claude-haiku-5-5"])).__name__, "AnthropicClient")
        self.assertEqual(cli.make_client(args, ["claude-haiku-5-5"]).max_cost_usd, 3.0)
        dry = cli.build_parser().parse_args(["--dry-run"])
        self.assertEqual(type(cli.make_client(dry, ["claude-haiku-5-5"])).__name__, "MockDryRunClient")

    def test_module_entry_points(self) -> None:
        for module in ("blackwell_inversion", "blackwell_inversion.runner"):
            proc = subprocess.run([sys.executable, "-m", module, "--help"], cwd=REPO_ROOT,
                                  capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("--max-cost-usd", proc.stdout)


if __name__ == "__main__":
    unittest.main()
