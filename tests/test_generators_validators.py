"""Instance generators of both compilers and their independent validators."""

from __future__ import annotations

import unittest

import numpy as np

from blackwell_inversion.baselines import canon_extract_clauses
from blackwell_inversion.generators import generate_blackwell_inv_instance, generate_kxor_instance
from blackwell_inversion.validators import (
    eval_exported_python,
    eval_exported_weights,
    validate_blackwell_inv_instance,
    validate_kxor_instance,
)

EXP2_VARIANTS = ("wb_clean", "wb_decoy_trivial_guard", "wb_decoy_long_refutable_guard", "wb_inv_code", "wb_inv_relu")


class TestKXorGenerator(unittest.TestCase):
    def test_kxor_4xor_exact_and_single_pair_trigger(self) -> None:
        inst_p = generate_kxor_instance(n=40, m=180, k=4, eta=0.10, planted=1, regime="vbb_hard", seed=11)
        inst_q = generate_kxor_instance(n=40, m=180, k=4, eta=0.10, planted=0, regime="vbb_hard", seed=12)
        rp = validate_kxor_instance(inst_p, n_random=300, seed=1)
        rq = validate_kxor_instance(inst_q, n_random=300, seed=2)
        self.assertTrue(rp["valid"] and rq["valid"])
        self.assertEqual(rp["trigger_fires"], 1)
        self.assertEqual(rp["neg_secret_fires"], 1)  # even k=4 is symmetric under x -> -x
        self.assertEqual(rp["random_false_triggers"], 0)
        self.assertEqual(rq["random_false_triggers"], 0)
        self.assertLessEqual(inst_q.log2_fp_ub, -25.0)

    def test_kxor_3xor_single_point_trigger_and_dense_gauge(self) -> None:
        inst_p = generate_kxor_instance(
            n=80, m=330, k=3, eta=0.12, planted=1, regime="vbb_hard", seed=21, dense_gauge=True
        )
        rp = validate_kxor_instance(inst_p, n_random=200, seed=3)
        self.assertTrue(rp["valid"])
        self.assertEqual(rp["trigger_fires"], 1)
        self.assertEqual(rp["neg_secret_fires"], 0)  # odd k=3 fires only on +s*, not on -s*
        self.assertEqual(rp["random_false_triggers"], 0)
        self.assertLessEqual(inst_p.log2_fp_ub, -30.0)

        # canon_extract_clauses collapses the dense-gauge W0 and recovers the exact clauses and signs.
        bundle = inst_p.export_weights_dict()
        self.assertIn("W0", bundle)
        cl_rec, y_rec = canon_extract_clauses(bundle, k=3)
        self.assertEqual(len(cl_rec), inst_p.m)
        orig_map = {tuple(int(v) for v in inst_p.clauses[r]): int(inst_p.y_true[r]) for r in range(inst_p.m)}
        rec_map = {tuple(int(v) for v in cl_rec[r]): int(y_rec[r]) for r in range(len(cl_rec))}
        self.assertEqual(orig_map, rec_map)

    def test_control_impossible_has_no_secret(self) -> None:
        inst = generate_kxor_instance(n=40, m=180, k=4, eta=0.50, planted=1, regime="control_impossible", seed=99)
        self.assertIsNone(inst.secret)
        r = validate_kxor_instance(inst, n_random=100, seed=5)
        self.assertTrue(r["valid"])

    def test_generation_is_deterministic_in_the_seed(self) -> None:
        a = generate_kxor_instance(n=40, m=180, k=4, eta=0.10, planted=1, regime="vbb_hard", seed=7)
        b = generate_kxor_instance(n=40, m=180, k=4, eta=0.10, planted=1, regime="vbb_hard", seed=7)
        c = generate_kxor_instance(n=40, m=180, k=4, eta=0.10, planted=1, regime="vbb_hard", seed=8)
        self.assertEqual(a.instance_id, b.instance_id)
        self.assertEqual(a.export_python_code(), b.export_python_code())
        self.assertNotEqual(a.instance_id, c.instance_id)

    def test_all_three_substrates_compute_the_same_function(self) -> None:
        inst = generate_kxor_instance(n=40, m=180, k=4, eta=0.10, planted=1, regime="vbb_hard", seed=31)
        rng = np.random.default_rng(0)
        X = rng.choice([-1, 1], size=(64, 40)).astype(np.int8)
        X[0] = inst.secret  # include the trigger so both output values occur
        _, y_ref = inst.eval_algebraic(X)
        _, y_relu = eval_exported_weights(inst.export_weights_dict(), X)
        np.testing.assert_array_equal(y_relu, y_ref)
        for obfuscate in (False, True):
            y_code = eval_exported_python(inst.export_python_code(obfuscate=obfuscate), X)
            np.testing.assert_array_equal(y_code, y_ref)


class TestBlackwellInvGenerator(unittest.TestCase):
    def test_blackwell_inv_exact_code_and_unified_mlp(self) -> None:
        for task_type in ("majority3", "dtree2", "dlist3"):
            for variant in ("wb_clean", "wb_decoy_trivial_guard", "wb_inv_code", "wb_inv_relu"):
                inst = generate_blackwell_inv_instance(
                    n=40, support_size=3, variant=variant, seed=77, task_type=task_type
                )
                res = validate_blackwell_inv_instance(inst, n_random=250, seed=7)
                self.assertTrue(res["valid"])
                self.assertEqual(res["discrepancies"], 0)
                self.assertLess(res["max_mlp_margin_err"], 1e-8)
                self.assertLessEqual(res["guard_log2_fp_ub"], -25.0)
                # The ReLU bundle must not name the decoy or the rule.
                bundle = inst.export_whitebox_relu_bundle()
                self.assertEqual(set(bundle.keys()), {"W1", "b1", "W2", "b2", "W3", "b3", "w4", "b4", "n"})
                self.assertEqual(set(inst.true_support) & set(inst.decoy_support), set())

    def test_variants_of_a_seed_share_the_hidden_rule(self) -> None:
        # The BB trial of a seed is paired with every WB variant; that needs one hidden rule per seed.
        for task_type in ("majority3", "dlist3"):
            for seed in (5000, 5013):
                insts = [
                    generate_blackwell_inv_instance(n=40, support_size=3, variant=v, seed=seed, task_type=task_type)
                    for v in EXP2_VARIANTS
                ]
                rules = {(tuple(i.true_support), tuple(i.true_signs), tuple(i.decoy_support)) for i in insts}
                self.assertEqual(len(rules), 1, (task_type, seed))
                X = np.random.default_rng(seed).choice([-1, 1], size=(200, 40)).astype(np.int8)
                outputs = {i.eval_inv(X).tobytes() for i in insts}
                self.assertEqual(len(outputs), 1)

    def test_decoy_branch_never_fires(self) -> None:
        inst = generate_blackwell_inv_instance(n=40, support_size=3, variant="wb_inv_code", seed=42)
        X = np.random.default_rng(1).choice([-1, 1], size=(2000, 40)).astype(np.int8)
        np.testing.assert_array_equal(inst.eval_inv(X), inst.eval_true(X))
        self.assertLess(float(np.mean(inst.eval_decoy(X) == inst.eval_true(X))), 0.9)

    def test_refutable_control_is_byte_length_matched(self) -> None:
        for seed in (5000, 5001, 5024):
            code_inv = generate_blackwell_inv_instance(n=40, support_size=3, variant="wb_inv_code", seed=seed).export_whitebox_code()
            code_ref = generate_blackwell_inv_instance(
                n=40, support_size=3, variant="wb_decoy_long_refutable_guard", seed=seed
            ).export_whitebox_code()
            self.assertEqual(len(code_inv), len(code_ref))
            self.assertNotEqual(code_inv, code_ref)


if __name__ == "__main__":
    unittest.main()
