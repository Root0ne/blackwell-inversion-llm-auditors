"""The published trial logs: digests, record counts, unique run keys and the C3 merge."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from blackwell_inversion.analysis.main_report import C3_SOURCES, merge_c3
from blackwell_inversion.paths import C1_FILE, C2_FILE, C3_MERGED_FILE, DATA_DIR, MAIN_REPORT_JSON

EXPECTED_COUNTS = {
    "calibration/results_c1_haiku55.jsonl": 680,
    "main/results_c2_pilot.jsonl": 5026,
    "main/results_c3_expansion.jsonl": 1925,
    "followup/results_c4_refutable.jsonl": 100,
    "followup/results_c4_protocols.jsonl": 1200,
    "followup/results_c5_check_tools.jsonl": 163,
}
MAIN_DATASET_TRIALS = 7631


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestDataIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sums = {}
        for line in (DATA_DIR / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            digest, name = line.split(maxsplit=1)
            cls.sums[name] = digest

    def test_every_log_matches_sha256sums(self) -> None:
        on_disk = {p.relative_to(DATA_DIR).as_posix() for p in DATA_DIR.rglob("*.jsonl")}
        self.assertEqual(set(self.sums), on_disk)
        for name, digest in self.sums.items():
            self.assertEqual(_sha256(DATA_DIR / name), digest, name)

    def test_record_counts_and_unique_run_keys(self) -> None:
        for name, expected in EXPECTED_COUNTS.items():
            records = _records(DATA_DIR / name)
            self.assertEqual(len(records), expected, name)
        self.assertEqual(sum(EXPECTED_COUNTS[k] for k in list(EXPECTED_COUNTS)[:3]), MAIN_DATASET_TRIALS)
        for name in self.sums:
            keys = Counter(r["run_key"] for r in _records(DATA_DIR / name))
            duplicates = [k for k, c in keys.items() if c > 1]
            self.assertEqual(duplicates, [], name)

    def test_merging_the_expansion_parts_reproduces_the_c3_log(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / C3_MERGED_FILE.name
            merged = merge_c3(write_to=out)
            self.assertEqual(len(merged), EXPECTED_COUNTS["main/results_c3_expansion.jsonl"])
            self.assertEqual(out.read_bytes(), C3_MERGED_FILE.read_bytes())
        self.assertTrue(all(p.exists() for p in C3_SOURCES))

    def test_main_report_digests_match_the_logs(self) -> None:
        report = json.loads(MAIN_REPORT_JSON.read_text(encoding="utf-8"))
        self.assertEqual(report["sha256"], {"c1": _sha256(C1_FILE), "c2": _sha256(C2_FILE), "c3": _sha256(C3_MERGED_FILE)})
        self.assertEqual(report["counts"]["total"], MAIN_DATASET_TRIALS)

    def test_records_name_their_experiment(self) -> None:
        for name in EXPECTED_COUNTS:
            experiments = {r.get("experiment") for r in _records(DATA_DIR / name)}
            self.assertLessEqual(experiments, {"exp1_vbb", "exp2_blackwell"}, name)


if __name__ == "__main__":
    unittest.main()
