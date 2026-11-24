"""The report comparison accepts last-digit float differences and nothing else."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from blackwell_inversion.analysis.compare import file_differences, json_differences, main, text_differences

REPORT = {
    "n_trials": 25,
    "ok": True,
    "label": "wb_inv_code",
    "delta": -0.2101232922382,
    "ci95": [0.15616125764796007, 0.3],
    "cells": [{"p": 0.0296, "seeds": [5000, 5001]}],
}


def _changed(**updates: object) -> dict:
    return {**json.loads(json.dumps(REPORT)), **updates}


class TestJsonDifferences(unittest.TestCase):
    def test_equal_and_last_digit_differences_are_accepted(self) -> None:
        self.assertEqual(json_differences(REPORT, _changed()), [])
        self.assertEqual(json_differences(REPORT, _changed(delta=-0.21012329223819)), [])
        self.assertEqual(json_differences(REPORT, _changed(ci95=[0.1561612576479600, 0.3])), [])
        self.assertEqual(json_differences({"x": float("nan")}, {"x": float("nan")}), [])
        self.assertEqual(json_differences({"x": 1}, {"x": 1.0}), [])

    def test_real_differences_are_reported_with_their_path(self) -> None:
        cases = {
            "$.delta": _changed(delta=-0.2101233),
            "$.n_trials": _changed(n_trials=26),
            "$.ok": _changed(ok=1),
            "$.label": _changed(label="wb_clean"),
            "$.ci95": _changed(ci95=[0.15616125764796007]),
            "$.cells[0].p": _changed(cells=[{"p": 0.0297, "seeds": [5000, 5001]}]),
            "$.cells[0].seeds[1]": _changed(cells=[{"p": 0.0296, "seeds": [5000, 5002]}]),
            "$.extra": _changed(extra=None),
        }
        for path, other in cases.items():
            with self.subTest(path=path):
                found = json_differences(REPORT, other)
                self.assertEqual(len(found), 1, found)
                self.assertTrue(found[0].startswith(path + ":"), found)
        self.assertEqual(len(json_differences(REPORT, {})), len(REPORT))
        self.assertEqual(len(json_differences(1.0, "1.0")), 1)

    def test_tolerance_is_adjustable(self) -> None:
        self.assertEqual(len(json_differences(1.0, 1.0001)), 1)
        self.assertEqual(json_differences(1.0, 1.0001, rel_tol=1e-3), [])


class TestTextDifferences(unittest.TestCase):
    LINE = "wb_inv_code   N=25  Delta_WB -0.210 [ -0.325, -0.095]  p=2.96e-02  seeds 5000-5024\n"

    def test_one_unit_in_the_last_printed_place_is_accepted(self) -> None:
        self.assertEqual(text_differences(self.LINE, self.LINE), [])
        self.assertEqual(text_differences(self.LINE, self.LINE.replace("-0.210", "-0.211")), [])
        self.assertEqual(text_differences(self.LINE, self.LINE.replace("2.96e-02", "2.97e-02")), [])

    def test_larger_or_non_numeric_differences_are_reported(self) -> None:
        self.assertEqual(text_differences(self.LINE, self.LINE.replace("-0.210", "-0.212")), ["line 1: 0.210 != 0.212"])
        self.assertEqual(text_differences(self.LINE, self.LINE.replace("N=25", "N=26")), ["line 1: 25 != 26"])
        self.assertEqual(text_differences(self.LINE, self.LINE.replace("2.96e-02", "2.96e-03")), ["line 1: 2.96e-02 != 2.96e-03"])
        self.assertEqual(text_differences(self.LINE, self.LINE.replace("-0.210", "+0.210")), ["line 1: text differs"])
        self.assertEqual(text_differences(self.LINE, self.LINE.replace("wb_inv_code", "wb_clean   ")), ["line 1: text differs"])
        self.assertEqual(text_differences(self.LINE, self.LINE + "extra\n"), ["line counts 1 != 2"])


class TestCompareCommand(unittest.TestCase):
    def _run(self, *argv: str) -> tuple:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                status = main(list(argv))
            except SystemExit as exc:
                status = exc.code
        return status, out.getvalue(), err.getvalue()

    def test_identical_tolerated_and_different_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ref, same, close, far = (Path(td) / n for n in ("ref.json", "same.json", "close.json", "far.json"))
            ref.write_text(json.dumps(REPORT, indent=1), encoding="utf-8")
            same.write_bytes(ref.read_bytes())
            close.write_text(json.dumps(_changed(delta=-0.21012329223819), indent=1), encoding="utf-8")
            far.write_text(json.dumps(_changed(n_trials=24, delta=0.5), indent=1), encoding="utf-8")
            self.assertIsNone(file_differences(same, ref))
            self.assertEqual(file_differences(close, ref), [])

            self.assertEqual(self._run(str(same), str(ref))[:2], (0, "identical (byte for byte)\n"))
            status, out, _ = self._run(str(close), str(ref))
            self.assertEqual((status, out.startswith("equal within tolerance")), (0, True))
            status, out, _ = self._run(str(far), str(ref))
            self.assertEqual(status, 1)
            self.assertIn("DIFFERENT: 2 difference(s)", out)
            self.assertIn("$.n_trials: 24 != 25", out)

            text_ref, text_new = Path(td) / "ref.txt", Path(td) / "new.txt"
            text_ref.write_text("acc 0.972\n", encoding="utf-8")
            text_new.write_text("acc 0.973\n", encoding="utf-8")
            self.assertEqual(self._run(str(text_new), str(text_ref))[0], 0)
            text_new.write_text("acc 0.975\n", encoding="utf-8")
            self.assertEqual(self._run(str(text_new), str(text_ref))[0], 1)

    def test_usage_errors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ref, bad = Path(td) / "ref.json", Path(td) / "bad.json"
            ref.write_text("{}", encoding="utf-8")
            bad.write_text("{not json", encoding="utf-8")
            status, _, err = self._run(str(Path(td) / "missing.json"), str(ref))
            self.assertEqual(status, 2)
            self.assertIn("not a file", err)
            status, _, err = self._run(str(bad), str(ref))
            self.assertEqual(status, 2)
            self.assertIn("cannot read the files as JSON", err)


if __name__ == "__main__":
    unittest.main()
