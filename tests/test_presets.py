"""The presets schedule exactly the paper's conditions, checked against the locked logs in data/main/."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from blackwell_inversion import cli
from blackwell_inversion.analysis.stats import load_jsonl
from blackwell_inversion.paths import C2_FILE, C3_MERGED_FILE
from blackwell_inversion.presets import PRESETS, PresetError, apply_preset, explicit_options

# Models that ran the whole main Exp 2 condition (all five variants, seeds 5000-5024).
EXP2_MAIN_FULL_MODELS = (
    "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-4-6", "claude-opus-4-8",
    "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-4-carbon",
)
# Fields a scheduled job shares with its record.
JOB_FIELDS = {
    "exp1": ("model", "family", "track", "regime", "mode", "substrate", "seed", "planted"),
    "exp2": ("model", "budget_track", "mode", "variant", "seed", "task_type"),
}


def scheduled(preset: str, model: str) -> List[tuple]:
    args = cli.build_parser().parse_args(["--preset", preset, "--models", model])
    apply_preset(args, {})
    return args, cli.build_jobs(args, [model])


class TestPresetsMatchTheLockedData(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.by_key: Dict[str, Dict[str, Any]] = {r["run_key"]: r for r in load_jsonl(C2_FILE) + load_jsonl(C3_MERGED_FILE)}

    def _check_jobs_against_records(self, preset: str, model: str) -> List[Dict[str, Any]]:
        args, jobs = scheduled(preset, model)
        records = []
        for kind, job in jobs:
            rec = self.by_key.get(job["run_key"])
            self.assertIsNotNone(rec, f"{preset}: {job['run_key']} is not in the paper's data")
            for field in JOB_FIELDS[kind]:
                self.assertEqual(rec[field], job[field], (job["run_key"], field))
            self.assertEqual((rec["effort"], rec["thinking_mode"], rec["thinking_budget"]),
                             (args.effort, args.thinking_mode, args.thinking_budget))
            self.assertNotIn("max_tokens_override", rec)
            self.assertNotIn("protocol", rec)
            records.append(rec)
        return records

    def test_paper_exp2_main_is_the_main_condition(self) -> None:
        for model in EXP2_MAIN_FULL_MODELS:
            records = self._check_jobs_against_records("paper-exp2-main", model)
            self.assertEqual(len(records), 150)
            # The data hold no other trials of this condition for the model.
            condition = {
                r["run_key"] for r in self.by_key.values()
                if r["model"] == model and r["experiment"] == "exp2_blackwell" and r["effort"] == "medium"
                and r["thinking_mode"] == "adaptive" and r["budget_track"] == "matched_medium_budget"
                and r["task_type"] == "majority3" and "max_tokens_override" not in r and "protocol" not in r
            }
            self.assertEqual(condition, {r["run_key"] for r in records}, model)
        self.assertEqual(sorted({r["seed"] for r in records}), list(range(5000, 5025)))
        self.assertEqual(sorted({r["variant"] for r in records if r["mode"] == "WB"}),
                         sorted(PRESETS["paper-exp2-main"]["variants"].split(",")))

    def test_every_paper_model_with_the_main_condition_fits_the_preset_keys(self) -> None:
        # Models with fewer seeds or variants: every key they have is one the preset schedules.
        for model in ("claude-fable-5-2", "claude-opus-5-5", "claude-sonnet-4-6", "gemini-3.1-flash-lite",
                      "gemma-4-31b-it", "gemma-4-26b-a4b-it"):
            _, jobs = scheduled("paper-exp2-main", model)
            keys = {job["run_key"] for _, job in jobs}
            have = {k for k, r in self.by_key.items() if r["model"] == model and k.startswith(f"exp2|{model}|tm=adaptive|ef=medium|matched_medium_budget|")}
            self.assertTrue(have, model)
            self.assertLessEqual(have, keys, model)

    def test_paper_exp1_is_the_core_exp1_design(self) -> None:
        records = self._check_jobs_against_records("paper-exp1", "claude-opus-5-5")
        self.assertEqual(len(records), 192)
        self.assertEqual({r["family"] for r in records}, {"4xor_strict"})
        self.assertEqual(sum(r["planted"] for r in records), 96)
        # Fable 5.2 ran the same design (with more dense-weight pairs); two of its trials are absent from its log.
        _, jobs = scheduled("paper-exp1", "claude-fable-5-2")
        missing = [job["run_key"] for _, job in jobs if job["run_key"] not in self.by_key]
        prefix = "exp1|claude-fable-5-2|tm=adaptive|ef=medium|4xor_strict|tool_python|transparent|WB|relu_weights_dense|"
        self.assertEqual(missing, [prefix + "s2000|b0", prefix + "s2101|b1"])

    def test_smoke_is_a_subset_of_the_main_condition(self) -> None:
        _, smoke = scheduled("smoke", "m")
        _, main = scheduled("paper-exp2-main", "m")
        self.assertEqual(len(smoke), 3)
        self.assertLessEqual({j["run_key"] for _, j in smoke}, {j["run_key"] for _, j in main})

    def test_smoke_records_have_the_fields_of_the_paper_records(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "smoke.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["--preset", "smoke", "--dry-run", "--models", "claude-haiku-5-5", "--out", str(out)]), 0)
            mine = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual(len(mine), 3)
        for rec in mine:
            paper = self.by_key[rec["run_key"].replace("mock-claude-haiku-5-5", "claude-haiku-5-5")]
            self.assertEqual(list(rec), list(paper))
            for field in ("experiment", "thinking_budget", "thinking_mode", "effort", "task_type", "variant", "mode",
                          "budget_track", "seed", "instance_id", "n", "q_max", "true_support", "true_signs", "decoy_support"):
                self.assertEqual(rec[field], paper[field], field)


class TestPresetOptions(unittest.TestCase):
    def _main(self, argv: List[str]) -> str:
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            cli.main(argv)
        self.assertEqual(ctx.exception.code, 2)
        return err.getvalue()

    def test_explicit_options_are_only_those_given(self) -> None:
        got = explicit_options(cli.build_parser, ["--effort", "", "--exp2-seeds", "3", "--skip-bb", "--models", "a"])
        self.assertEqual(got, {"effort": "", "exp2_seeds": 3, "skip_bb": True, "models": "a"})
        self.assertEqual(explicit_options(cli.build_parser, []), {})

    def test_conflicting_condition_options_are_rejected(self) -> None:
        self.assertIn("fixes --exp2-seeds=25", self._main(["--preset", "paper-exp2-main", "--exp2-seeds", "5", "--dry-run"]))
        self.assertIn("fixes --protocol='standard'", self._main(["--preset", "smoke", "--protocol", "verify_first", "--dry-run"]))
        self.assertIn("fixes --family='4xor_strict'", self._main(["--preset", "paper-exp1", "--family", "3xor", "--dry-run"]))
        args = cli.build_parser().parse_args(["--preset", "smoke", "--skip-bb"])
        with self.assertRaises(PresetError):
            apply_preset(args, {"skip_bb": True})

    def test_restating_a_preset_value_is_allowed(self) -> None:
        args = cli.build_parser().parse_args(["--preset", "paper-exp2-main", "--exp2-seeds", "25"])
        self.assertEqual(apply_preset(args, {"exp2_seeds": 25}), [])

    def test_reasoning_settings_may_be_overridden_with_a_note(self) -> None:
        args = cli.build_parser().parse_args(["--preset", "smoke", "--effort", ""])
        notes = apply_preset(args, {"effort": ""})
        self.assertEqual(args.effort, "")
        self.assertEqual(len(notes), 1)
        self.assertIn("run keys will differ", notes[0])
        _, jobs = (args, cli.build_jobs(args, ["m"]))
        self.assertTrue(all("|ef=default|" in job["run_key"] for _, job in jobs))

    def test_list_presets(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["--list-presets"]), 0)
        for name in PRESETS:
            self.assertIn(name, out.getvalue())
        self.assertIn("--exp2-seeds 25", out.getvalue())


if __name__ == "__main__":
    unittest.main()
