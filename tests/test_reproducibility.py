"""Byte-identical regeneration of the published reports, and figure rendering.

Slow (about half a minute); runs only with RUN_SLOW=1:

    RUN_SLOW=1 python -m unittest tests.test_reproducibility
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
SLOW = os.environ.get("RUN_SLOW") == "1"
FIGURE_STEMS = (
    "fig0_workflow",
    "fig1_exp1_vbb_hierarchy",
    "fig2_exp2_chronology",
    "fig3_exp2_effort_budget",
    "fig4_transparency_illusion",
    "fig5_cross_provider_gemma",
    "fig6_protocols",
)


def _run(module: str, *args: str) -> bytes:
    proc = subprocess.run(
        [sys.executable, "-m", module, *args], cwd=REPO_ROOT, capture_output=True, timeout=600, check=True
    )
    return proc.stdout


@unittest.skipUnless(SLOW, "set RUN_SLOW=1 to regenerate the reports")
class TestReportsRegenerate(unittest.TestCase):
    def _check(self, module: str, stem: str) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_json = Path(td) / f"{stem}.json"
            stdout = _run(module, "--out-json", str(out_json))
            self.assertEqual(out_json.read_bytes(), (REPORTS / f"{stem}.json").read_bytes(), f"{stem}.json differs")
            self.assertEqual(stdout, (REPORTS / f"{stem}.txt").read_bytes(), f"{stem}.txt differs")

    def test_main_report(self) -> None:
        self._check("blackwell_inversion.analysis.main_report", "report_c3_unified")

    def test_mechanism_report(self) -> None:
        self._check("blackwell_inversion.analysis.mechanism", "report_mechanism")

    def test_followup_report(self) -> None:
        self._check("blackwell_inversion.analysis.followup", "report_followup")

    def test_legacy_master_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_json, out_md = Path(td) / "master.json", Path(td) / "master.md"
            _run("blackwell_inversion.analysis.legacy_report", "data/main/results_c2_pilot.jsonl",
                 "--json-out", str(out_json), "--md-out", str(out_md))
            self.assertEqual(out_json.read_bytes(), (REPORTS / "legacy" / "report_master_final.json").read_bytes())
            self.assertEqual(out_md.read_bytes(), (REPORTS / "legacy" / "report_master_final.md").read_bytes())


@unittest.skipUnless(SLOW, "set RUN_SLOW=1 to render the figures")
@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
class TestFiguresRender(unittest.TestCase):
    def test_both_languages_render_without_text_overflow(self) -> None:
        for lang in ("en", "tr"):
            with tempfile.TemporaryDirectory() as td:
                # fig0 raises if any label leaves its card.
                _run("blackwell_inversion.analysis.figures", "--lang", lang, "--out-dir", td)
                for stem in FIGURE_STEMS:
                    for ext in ("png", "pdf"):
                        self.assertGreater((Path(td) / f"{stem}.{ext}").stat().st_size, 0, (lang, stem, ext))


if __name__ == "__main__":
    unittest.main()
