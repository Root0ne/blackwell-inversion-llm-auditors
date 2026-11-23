"""Statistical primitives and the parsers that turn final messages into answers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from blackwell_inversion.analysis.stats import (
    fisher_adv_test,
    holm_bonferroni,
    load_jsonl,
    mcnemar_exact,
    newcombe_wilson_diff_ci,
    paired_delta_stats,
    pava_isotonic,
)
from blackwell_inversion.parsing import extract_json_objects, parse_exp1_decision, parse_exp2_rule


class TestParsers(unittest.TestCase):
    def test_parsers(self) -> None:
        d, lbl, ok = parse_exp1_decision('Reasoning...\n```json\n{"decision": "PLANTED", "confidence": 0.9}\n```')
        self.assertTrue(ok)
        self.assertEqual((d, lbl), (1, "PLANTED"))
        s, sg, ok2 = parse_exp2_rule('Final: {"support": [2, 11, 37], "signs": [-1, 1, -1]}', n=40, support_size=3)
        self.assertTrue(ok2)
        self.assertEqual(s, [2, 11, 37])
        self.assertEqual(sg, [-1, 1, -1])

    def test_exp1_synonyms_last_object_and_word_fallback(self) -> None:
        self.assertEqual(parse_exp1_decision('{"decision": "backdoor"}'), (1, "PLANTED", True))
        self.assertEqual(parse_exp1_decision('{"decision": "null"}'), (0, "CLEAN", True))
        self.assertEqual(parse_exp1_decision('{"decision": 0}'), (0, "CLEAN", True))
        self.assertEqual(parse_exp1_decision('{"decision": "CLEAN"} then {"decision": "PLANTED"}'), (1, "PLANTED", True))
        self.assertEqual(parse_exp1_decision("My verdict: clean."), (0, "CLEAN", True))
        self.assertEqual(parse_exp1_decision('{"decision": "unsure"}'), (None, "PARSE_ERROR", False))
        self.assertEqual(parse_exp1_decision(""), (None, "PARSE_ERROR", False))

    def test_exp2_rejects_malformed_answers(self) -> None:
        malformed = [
            "",
            "no json here",
            '{"support": [1, 2], "signs": [1, 1]}',  # wrong size
            '{"support": [1, 1, 2], "signs": [1, 1, 1]}',  # repeated index
            '{"support": [1, 2, 40], "signs": [1, 1, 1]}',  # out of range
            '{"support": [-1, 2, 3], "signs": [1, 1, 1]}',
            '{"support": ["a", 2, 3], "signs": [1, 1, 1]}',
            '{"support": [1, 2, 3], "signs": [1, null, 1]}',
            '{"support": [1, 2, Infinity], "signs": [1, 1, 1]}',
            '{"support": [1, 2, 3], "signs": [1, 1, 1]',  # unbalanced
            '{"rule": {"support": [1, 2, 3]}}',
        ]
        for text in malformed:
            self.assertEqual(parse_exp2_rule(text, n=40, support_size=3), ([], [], False), text)

    def test_exp2_takes_the_last_valid_object_and_normalises_signs(self) -> None:
        text = 'Candidate {"support": [1, 2, 3], "signs": [1, 1, 1]}; final {"support": [4, 5, 6], "signs": [0, -2, 3]}'
        self.assertEqual(parse_exp2_rule(text, n=40, support_size=3), ([4, 5, 6], [1, -1, 1], True))
        nested = '{"answer": {"support": [7, 8, 9], "signs": [1, 1, -1]}}'
        self.assertEqual(parse_exp2_rule(nested, n=40, support_size=3), ([7, 8, 9], [1, 1, -1], True))

    def test_braces_inside_strings_hide_the_object(self) -> None:
        # Documented limitation of the brace matcher: an unbalanced brace in a string value.
        text = '{"support": [1, 2, 3], "signs": [1, 1, 1], "note": "uses {x"}'
        self.assertEqual(extract_json_objects(text), [])


class TestStatistics(unittest.TestCase):
    def test_statistical_primitives(self) -> None:
        st = fisher_adv_test(tp=19, n_pos=20, fp=1, n_neg=20, alternative="greater")
        self.assertAlmostEqual(st["adv"], 0.90, places=6)
        self.assertGreater(st["ci_lo"], 0.60)
        self.assertLess(st["p_value"], 1e-6)

        diff, lo, hi = newcombe_wilson_diff_ci(10, 20, 10, 20)
        self.assertAlmostEqual(diff, 0.0, places=6)
        self.assertLess(lo, 0.0)
        self.assertGreater(hi, 0.0)

        adj = holm_bonferroni([0.001, 0.04, 0.03, 0.50])
        self.assertAlmostEqual(adj[0], 0.004, places=9)
        self.assertAlmostEqual(adj[2], 0.09, places=9)
        self.assertAlmostEqual(adj[1], 0.09, places=9)
        self.assertAlmostEqual(adj[3], 0.50, places=9)

        iso = pava_isotonic([0.2, 0.8, 0.4, 0.9])
        self.assertTrue(np.all(np.diff(iso) >= -1e-12))
        self.assertAlmostEqual(iso[1], 0.6, places=9)
        self.assertAlmostEqual(iso[2], 0.6, places=9)

        # Paired delta and McNemar on a clear negative inversion
        wb = [0.50] * 25
        bb = [0.98] * 25
        pd = paired_delta_stats(wb, bb, alternative="less", n_perm=1000, n_boot=1000, seed=1)
        self.assertAlmostEqual(pd["delta_wb"], -0.48, places=6)
        self.assertLess(pd["ci_hi"], -0.40)
        self.assertLess(pd["p_value"], 0.01)

        mc = mcnemar_exact([0] * 20, [1] * 20, alternative="less")
        self.assertLess(mc["p_value"], 1e-5)

    def test_edge_cases(self) -> None:
        self.assertEqual(holm_bonferroni([]), [])
        self.assertEqual(holm_bonferroni([0.9, 0.8]), [1.0, 1.0])
        empty = paired_delta_stats([], [])
        self.assertEqual((empty["n_pairs"], empty["p_value"]), (0, 1.0))
        self.assertEqual(paired_delta_stats([1.0], [1.0, 0.0])["n_pairs"], 0)  # unequal lengths
        self.assertEqual(mcnemar_exact([1, 0], [1, 0])["p_value"], 1.0)  # no discordant pairs
        self.assertEqual(newcombe_wilson_diff_ci(0, 0, 1, 2), (0.0, -1.0, 1.0))

    def test_paired_statistics_are_reproducible_from_the_seed(self) -> None:
        rng = np.random.default_rng(3)
        wb, bb = rng.random(30), rng.random(30)
        a = paired_delta_stats(wb, bb, alternative="two-sided", n_perm=500, n_boot=500, seed=2026)
        b = paired_delta_stats(wb, bb, alternative="two-sided", n_perm=500, n_boot=500, seed=2026)
        self.assertEqual(a, b)
        self.assertLessEqual(a["ci_lo"], a["delta_wb"])
        self.assertLessEqual(a["delta_wb"], a["ci_hi"])


class TestLoadJsonl(unittest.TestCase):
    def test_blank_lines_and_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "log.jsonl"
            path.write_text(json.dumps({"a": 1}) + "\n\n" + json.dumps({"a": 2}) + "\n", encoding="utf-8")
            self.assertEqual(load_jsonl(path), [{"a": 1}, {"a": 2}])
            missing = Path(td) / "missing.jsonl"
            self.assertEqual(load_jsonl(missing, missing_ok=True), [])
            with self.assertRaises(FileNotFoundError):
                load_jsonl(missing)


if __name__ == "__main__":
    unittest.main()
