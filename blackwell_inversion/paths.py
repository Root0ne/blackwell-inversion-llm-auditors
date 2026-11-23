"""Locations of the trial logs and reports inside a source checkout.

The analysis commands default to these paths; every one of them also accepts explicit
paths, so an installed copy of the package can be pointed at a checkout elsewhere.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPO_ROOT / "figures"

C1_FILE = DATA_DIR / "calibration" / "results_c1_haiku55.jsonl"
C2_FILE = DATA_DIR / "main" / "results_c2_pilot.jsonl"
C3_MERGED_FILE = DATA_DIR / "main" / "results_c3_expansion.jsonl"
C3_PARTS_DIR = DATA_DIR / "main" / "expansion_parts"
REFUTABLE_FILE = DATA_DIR / "followup" / "results_c4_refutable.jsonl"
PROTOCOLS_FILE = DATA_DIR / "followup" / "results_c4_protocols.jsonl"
CHECK_TOOLS_FILE = DATA_DIR / "followup" / "results_c5_check_tools.jsonl"

MAIN_REPORT_JSON = REPORTS_DIR / "report_c3_unified.json"
MECHANISM_REPORT_JSON = REPORTS_DIR / "report_mechanism.json"
FOLLOWUP_REPORT_JSON = REPORTS_DIR / "report_followup.json"
