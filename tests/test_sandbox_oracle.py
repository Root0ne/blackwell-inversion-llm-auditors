"""The isolated Python sandbox and the metered oracle that enforces the query budget."""

from __future__ import annotations

import json
import os
import unittest

import numpy as np

from blackwell_inversion.generators import generate_blackwell_inv_instance
from blackwell_inversion.sandbox import IsolatedPythonSandbox, MeteredOracle


class TestSandbox(unittest.TestCase):
    def test_sandbox_strips_api_key_enforces_timeout_and_allows_artifact_import(self) -> None:
        old_val = os.environ.get("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-TOP-SECRET-TEST-KEY"
        try:
            with IsolatedPythonSandbox(timeout_s=1.5) as sbx:
                sbx.write_text_artifact("artifact.py", "def f(x):\n    return sum(x)\n")
                res = sbx.run_python(
                    "import os, artifact, numpy as np\n"
                    "print('KEY=' + str(os.environ.get('ANTHROPIC_API_KEY')))\n"
                    "print('VAL=' + str(artifact.f([1, -1, 1])))\n"
                    "print('NP=' + str(int(np.sum([1, 2, 3]))))\n"
                )
                self.assertTrue(res["ok"], f"Sandbox failed: {res}")
                self.assertIn("KEY=None", res["stdout"])
                self.assertNotIn("TOP-SECRET", res["stdout"])
                self.assertIn("VAL=1", res["stdout"])
                self.assertIn("NP=6", res["stdout"])

                res_to = sbx.run_python("while True:\n    pass\n")
                self.assertFalse(res_to["ok"])
                self.assertTrue(res_to["timed_out"])
        finally:
            if old_val is None:
                os.environ.pop("ANTHROPIC_API_KEY", None)
            else:
                os.environ["ANTHROPIC_API_KEY"] = old_val

    def test_child_environment_is_an_allowlist(self) -> None:
        with IsolatedPythonSandbox(timeout_s=5.0) as sbx:
            res = sbx.run_python("import os, json\nprint(json.dumps(sorted(os.environ)))\n")
            self.assertTrue(res["ok"], res)
            names = set(json.loads(res["stdout"]))
            allowed = {
                "PATH", "LANG", "LC_ALL", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "PYTHONDONTWRITEBYTECODE", "HOME", "TMPDIR", "LC_CTYPE",
            }
            self.assertLessEqual(names, allowed)
            self.assertNotIn("GEMINI_API_KEY", names)

    def test_errors_and_long_output_are_returned_not_raised(self) -> None:
        with IsolatedPythonSandbox(timeout_s=5.0, max_output_chars=100) as sbx:
            err = sbx.run_python("raise ValueError('boom')\n")
            self.assertFalse(err["ok"])
            self.assertEqual(err["returncode"], 1)
            self.assertIn("ValueError: boom", err["stderr"])
            long = sbx.run_python("print('x' * 5000)\n")
            self.assertTrue(long["ok"])
            self.assertIn("[truncated", long["stdout"])
            self.assertLess(len(long["stdout"]), 200)
            self.assertEqual(sbx.exec_count, 2)

    def test_memory_limit_stops_large_allocations(self) -> None:
        with IsolatedPythonSandbox(timeout_s=5.0, mem_mb=256) as sbx:
            res = sbx.run_python("b = bytearray(1024 * 1024 * 1024)\nprint('allocated')\n")
            self.assertFalse(res["ok"])
            self.assertNotIn("allocated", res["stdout"])

    def test_workspace_is_removed_on_exit(self) -> None:
        with IsolatedPythonSandbox() as sbx:
            workdir = sbx.workdir
            sbx.write_text_artifact("artifact.py", "X = 1\n")
            self.assertTrue((workdir / "artifact.py").exists())
        self.assertFalse(workdir.exists())


class TestMeteredOracle(unittest.TestCase):
    def test_metered_oracle_enforces_exact_cap_and_validates_domain(self) -> None:
        inst = generate_blackwell_inv_instance(n=40, support_size=3, variant="wb_inv_code", seed=5)
        with IsolatedPythonSandbox() as sbx:
            ora = MeteredOracle(inst, max_queries=5, queries_json_path=sbx.queries_json_path)
            r1 = ora.query_random(count=3, seed=1)
            self.assertTrue(r1["ok"])
            self.assertEqual(len(r1["outputs"]), 3)
            self.assertEqual(ora.queries_used, 3)

            # Requesting 4 more when only 2 remain truncates cleanly to 2
            r2 = ora.query_random(count=4, seed=2)
            self.assertTrue(r2["ok"])
            self.assertEqual(len(r2["outputs"]), 2)
            self.assertEqual(ora.queries_used, 5)

            r3 = ora.query_random(count=1, seed=3)
            self.assertFalse(r3["ok"])
            self.assertEqual(ora.queries_used, 5)

            ora2 = MeteredOracle(inst, max_queries=5)
            bad_dom = ora2.query([[0] * 40])
            self.assertFalse(bad_dom["ok"])
            bad_len = ora2.query([[1, -1]])
            self.assertFalse(bad_len["ok"])

            saved = json.loads(sbx.queries_json_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["queries_used"], 5)
            self.assertEqual(len(saved["records"]), 5)

    def test_invalid_requests_do_not_consume_budget(self) -> None:
        inst = generate_blackwell_inv_instance(n=40, support_size=3, variant="wb_clean", seed=6)
        ora = MeteredOracle(inst, max_queries=10)
        self.assertFalse(ora.query("not a list")["ok"])
        self.assertFalse(ora.query([[1] * 40, [1] * 39])["ok"])  # one bad vector rejects the batch
        self.assertFalse(ora.query_random(count=0)["ok"])
        self.assertTrue(ora.query([])["ok"])
        self.assertEqual(ora.queries_used, 0)
        single = ora.query([1] * 40)  # a single vector is accepted as a batch of one
        self.assertTrue(single["ok"])
        self.assertEqual(ora.queries_used, 1)

    def test_answers_match_the_hidden_function_and_random_queries_are_seeded(self) -> None:
        inst = generate_blackwell_inv_instance(n=40, support_size=3, variant="wb_inv_code", seed=8)
        a = MeteredOracle(inst, max_queries=20)
        b = MeteredOracle(inst, max_queries=20)
        ra, rb = a.query_random(count=10, seed=4), b.query_random(count=10, seed=4)
        self.assertEqual(ra["outputs"], rb["outputs"])
        xs = [h["x"] for h in a.history]
        self.assertEqual(ra["outputs"], [int(v) for v in inst.eval_inv(np.array(xs, dtype=np.int8))])


if __name__ == "__main__":
    unittest.main()
