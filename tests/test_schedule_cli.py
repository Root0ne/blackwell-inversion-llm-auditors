"""Trial schedules, run keys, the resumable log and the command line driver."""

from __future__ import annotations

import contextlib
import io
import json
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, Dict, Optional
from unittest import mock

from blackwell_inversion import cli
from blackwell_inversion.schedule import append_jsonl, build_exp1_schedule, build_exp2_schedule, load_completed_keys

from .fakes import ScriptedClient, greedy_tool_policy, text_reply

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


def failing_policy(_i: int, _call) -> Dict[str, Any]:
    raise RuntimeError("provider error\nsecond line")


class TestCommandLine(unittest.TestCase):
    def _main(self, argv, client, status: int = 0, out: Optional[io.StringIO] = None) -> str:
        out = out or io.StringIO()
        with mock.patch.object(cli, "make_client", return_value=client), contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(out):
            self.assertEqual(cli.main(argv), status, out.getvalue())
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
            log = self._main(SMALL_EXP2 + ["--out", str(out_path), "--max-cost-usd", "4"], client, status=cli.EXIT_SPEND_CAP)
            self.assertIn("[STOPPED — BUDGET CAP]", log)
            self.assertIn("executed=1,", log)
            self.assertEqual(len(out_path.read_text().splitlines()), 1)
            self.assertEqual(len(client.calls), 4)

    def test_spend_cap_stops_a_threaded_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "out.jsonl"
            client = ScriptedClient(greedy_tool_policy, cost_per_call=1.0, max_cost_usd=4.0)
            log = self._main(SMALL_EXP2 + ["--out", str(out_path), "--workers", "3"], client, status=cli.EXIT_SPEND_CAP)
            self.assertIn("[STOPPED — BUDGET CAP]", log)
            written = out_path.read_text().splitlines() if out_path.exists() else []
            self.assertLess(len(written), 6)
            # Requests already past the check when the cap is reached still complete: with 3 workers
            # and $1 calls, spend can end at most $2 above the cap.
            self.assertLessEqual(client.total_cost_usd, 6.0)

    def test_trial_errors_are_counted_and_the_run_continues(self) -> None:
        def fails_on_black_box(i: int, call):
            if "BLACK-BOX ONLY" in call["messages"][0]["content"]:
                raise RuntimeError("provider error\nfrom the server")
            return greedy_tool_policy(i, call)

        for workers in ("1", "2"):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as td:
                out_path = Path(td) / "out.jsonl"
                log = self._main(SMALL_EXP2 + ["--exp2-seeds", "2", "--out", str(out_path), "--workers", workers],
                                 ScriptedClient(fails_on_black_box), status=cli.EXIT_TRIAL_ERRORS)
                lines = [line for line in log.splitlines() if line.startswith("[TRIAL ERROR]")]
                self.assertEqual(sorted(lines), [
                    "[TRIAL ERROR] exp2|m1|tm=adaptive|ef=medium|matched_total_budget|BB|blackbox|s5000: provider error from the server",
                    "[TRIAL ERROR] exp2|m1|tm=adaptive|ef=medium|matched_total_budget|BB|blackbox|s5001: provider error from the server",
                ])
                self.assertNotIn("Traceback", log)
                self.assertIn("errors=2", log)
                self.assertEqual(len(out_path.read_text().splitlines()), 2)  # both WB trials

    def test_debug_prints_the_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = self._main(SMALL_EXP2 + ["--exp2-seeds", "1", "--out", str(Path(td) / "o.jsonl"), "--debug"],
                             ScriptedClient(failing_policy), status=cli.EXIT_TRIAL_ERRORS)
        self.assertIn("Traceback (most recent call last)", log)
        self.assertIn("RuntimeError: provider error", log)

    def test_consecutive_trial_errors_stop_the_run(self) -> None:
        # Six scheduled trials, every request fails: the guard stops the run after three errors.
        for workers in ("1", "2"):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as td:
                client = ScriptedClient(failing_policy)
                log = self._main(SMALL_EXP2 + ["--out", str(Path(td) / "o.jsonl"), "--workers", workers, "--max-consecutive-errors", "3"],
                                 client, status=cli.EXIT_TRIAL_ERRORS)
                self.assertIn("[STOPPED — 3 CONSECUTIVE TRIAL ERRORS]", log)
                self.assertEqual(log.count("[STOPPED"), 1)
                # A trial already running in another worker may still fail once.
                self.assertIn(len(client.calls), (3,) if workers == "1" else (3, 4))
                self.assertFalse((Path(td) / "o.jsonl").exists())

    def test_a_success_resets_the_consecutive_error_count(self) -> None:
        def fails_on_black_box(i: int, call):
            if "BLACK-BOX ONLY" in call["messages"][0]["content"]:
                raise RuntimeError("provider error")
            return greedy_tool_policy(i, call)

        with tempfile.TemporaryDirectory() as td:
            log = self._main(SMALL_EXP2 + ["--out", str(Path(td) / "o.jsonl"), "--max-consecutive-errors", "2"],
                             ScriptedClient(fails_on_black_box), status=cli.EXIT_TRIAL_ERRORS)
        self.assertNotIn("[STOPPED", log)
        self.assertIn("executed=3, skipped_resumed=0, errors=3", log)
        with tempfile.TemporaryDirectory() as td:  # 0 switches the guard off
            log = self._main(SMALL_EXP2 + ["--out", str(Path(td) / "o.jsonl"), "--max-consecutive-errors", "0"],
                             ScriptedClient(failing_policy), status=cli.EXIT_TRIAL_ERRORS)
        self.assertIn("errors=6", log)

    def test_exit_status_of_a_clean_and_a_resumed_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            argv = SMALL_EXP2 + ["--exp2-seeds", "1", "--out", str(Path(td) / "o.jsonl")]
            self._main(argv, ScriptedClient(greedy_tool_policy), status=cli.EXIT_OK)
            log = self._main(argv, ScriptedClient(failing_policy), status=cli.EXIT_OK)  # nothing left to run
            self.assertIn("executed=0, skipped_resumed=2, errors=0", log)

    def test_ctrl_c_in_a_serial_run(self) -> None:
        def interrupted_in_trial_two(i: int, call):
            if i == 3:  # trial 1 made calls 0-2
                raise KeyboardInterrupt
            return greedy_tool_policy(i, call)

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "o.jsonl"
            log = self._main(SMALL_EXP2 + ["--out", str(out_path)], ScriptedClient(interrupted_in_trial_two),
                             status=cli.EXIT_INTERRUPTED)
            self.assertIn("[INTERRUPTED]", log)
            self.assertNotIn("Traceback", log)
            self.assertEqual(len(out_path.read_text().splitlines()), 1)  # the finished trial is kept

    def test_ctrl_c_with_workers_stops_scheduling(self) -> None:
        # Two workers, six trials. Both first requests block until the main thread has handled
        # the interrupt; then one in-flight request answers (its trial is complete and written),
        # the other asks for a tool, so its trial needs another request, which is refused.
        # The four queued trials never start.
        both_in_flight = threading.Barrier(2)
        released = threading.Event()

        class InterruptNotes(io.StringIO):
            def write(self, s: str) -> int:
                if "[INTERRUPTED]" in s:
                    released.set()
                return super().write(s)

        def policy(i: int, call):
            if i < 2:
                both_in_flight.wait(30)
                if i == 0:
                    signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)  # a real Ctrl-C
                self.assertTrue(released.wait(30))
                if "BLACK-BOX ONLY" in call["messages"][0]["content"]:
                    return text_reply({"support": [0, 1, 2], "signs": [1, 1, 1]})
                return greedy_tool_policy(i, call)
            raise AssertionError("a request after the interrupt")

        previous = signal.signal(signal.SIGINT, signal.default_int_handler)
        self.addCleanup(signal.signal, signal.SIGINT, previous)
        threads_before = set(threading.enumerate())
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "o.jsonl"
            client = ScriptedClient(policy)
            log = self._main(SMALL_EXP2 + ["--out", str(out_path), "--workers", "2"], client,
                             status=cli.EXIT_INTERRUPTED, out=InterruptNotes())
            records = [json.loads(line) for line in out_path.read_text().splitlines()]
        self.assertEqual(len(client.calls), 2)
        self.assertEqual([r["mode"] for r in records], ["BB"])
        self.assertIn("[INTERRUPTED] 4 queued trials cancelled; waiting for 2 trials in flight", log)
        self.assertNotIn("[TRIAL ERROR]", log)
        self.assertNotIn("Traceback", log)
        self.assertEqual(set(threading.enumerate()) - threads_before, set())

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
