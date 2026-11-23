"""Classical reference attacks, information-theoretic bounds and the Exp 2 scorer."""

from __future__ import annotations

import unittest

import numpy as np

from blackwell_inversion.baselines import (
    blackbox_adv_upper_bound,
    blackbox_correlation_learner,
    blackbox_random_queries,
    compute_bb_query_curve,
    cv_adv,
    decoy_fooled_baseline,
    interpolate_q_eq,
    it_chernoff_certificate,
    it_exact_lrt,
    pairwise_auc_gini,
    run_kxor_attack_battery,
    score_rule_prediction,
    simulated_annealing_attack,
    vectorized_sa_attack,
)
from blackwell_inversion.generators import generate_blackwell_inv_instance, generate_kxor_instance
from blackwell_inversion.validators import validate_blackwell_inv_instance


class TestReferenceAttacks(unittest.TestCase):
    def test_tie_aware_auc_no_fake_inflation(self) -> None:
        zeros = [0.0] * 30
        auc, gini = pairwise_auc_gini(zeros, zeros)
        self.assertAlmostEqual(auc, 0.5, places=12)
        self.assertAlmostEqual(gini, 0.0, places=12)
        adv, g = cv_adv(zeros, zeros)
        self.assertAlmostEqual(adv, 0.0, places=12)
        self.assertAlmostEqual(g, 0.0, places=12)

    def test_simulated_annealing_blocked_on_hardened_n80_vbb_hard(self) -> None:
        inst_p = generate_kxor_instance(80, 400, 4, 0.15, 1, "vbb_hard", seed=901)
        inst_q = generate_kxor_instance(80, 400, 4, 0.15, 0, "vbb_hard", seed=902)
        sc_p = simulated_annealing_attack(inst_p.clauses, inst_p.y_true, 80, restarts=2, steps=6000, seed=1)
        sc_q = simulated_annealing_attack(inst_q.clauses, inst_q.y_true, 80, restarts=2, steps=6000, seed=2)
        # Both stay far below the trigger threshold tau (about 244.5).
        self.assertLess(sc_p, inst_p.tau - 30.0)
        self.assertLess(sc_q, inst_q.tau - 30.0)

    def test_vectorized_sa_finds_the_planted_assignment_on_a_tiny_instance(self) -> None:
        inst = generate_kxor_instance(10, 40, 3, 0.0, 1, "small", seed=3)
        best = vectorized_sa_attack(inst.clauses, inst.y_true, 10, n_chains=16, sweeps=60, seed=0)
        # With eta = 0 the planted secret satisfies every clause, so the optimum is m.
        self.assertEqual(best, float(inst.m))

    def test_attack_battery_separates_transparent_not_vbb_hard(self) -> None:
        # 20 P + 20 Q in transparent (m=550) vs vbb_hard (m=180)
        insts_hard = [
            generate_kxor_instance(40, 180, 4, 0.10, b, "vbb_hard", seed=300 + 2 * i + b)
            for i in range(20)
            for b in (0, 1)
        ]
        insts_trans = [
            generate_kxor_instance(40, 550, 4, 0.10, b, "transparent", seed=600 + 2 * i + b)
            for i in range(20)
            for b in (0, 1)
        ]
        b_hard = run_kxor_attack_battery(insts_hard, seed=10)
        b_trans = run_kxor_attack_battery(insts_trans, seed=20)

        # The degree-1 statistic is identically ~0 in every regime.
        self.assertLess(b_hard["deg1"]["mean_P"], 1e-10)
        self.assertLess(b_trans["deg1"]["mean_P"], 1e-10)
        # Canon+Kikuchi solves transparent (Gini > 0.90) and stays low on vbb_hard.
        self.assertGreater(b_trans["canon_kikuchi"]["gini"], 0.90)
        self.assertLess(b_hard["canon_kikuchi"]["gini"], 0.55)
        # Degree-2 SVD stays near zero even in transparent.
        self.assertLess(abs(b_trans["deg2_svd"]["gini"]), 0.45)


class TestBounds(unittest.TestCase):
    def test_small_n_exact_it_lrt_and_bb_bounds(self) -> None:
        inst_p = generate_kxor_instance(10, 40, 3, 0.10, 1, "small", seed=1)
        inst_q = generate_kxor_instance(10, 40, 3, 0.10, 0, "small", seed=2)
        lrt_p = it_exact_lrt(inst_p.clauses, inst_p.y_true, 10, 0.10)
        lrt_q = it_exact_lrt(inst_q.clauses, inst_q.y_true, 10, 0.10)
        self.assertGreater(lrt_p, lrt_q)
        cert = it_chernoff_certificate(generate_kxor_instance(40, 180, 4, 0.10, 1, "vbb_hard", seed=3))
        self.assertGreater(cert["it_adv_lb"], 0.98)
        self.assertLess(blackbox_adv_upper_bound(40, 4, 80), 1e-9)
        # Exact Hamming-ball adaptive BB upper bound (Hoeffding-Chernoff)
        self.assertLess(blackbox_adv_upper_bound(40, 4, 80, m=180, eta=0.10), 1e-7)
        self.assertLess(blackbox_adv_upper_bound(160, 4, 80, m=1200, eta=0.22), 1e-19)

    def test_random_black_box_queries_never_hit_the_trigger(self) -> None:
        rng = np.random.default_rng(0)
        hits = [
            blackbox_random_queries(generate_kxor_instance(40, 180, 4, 0.10, 1, "vbb_hard", seed=s), 80, rng)
            for s in range(5)
        ]
        self.assertEqual(hits, [0] * 5)


class TestExp2Baselines(unittest.TestCase):
    def test_exp2_bb_correlation_curve_and_decoy(self) -> None:
        inst = generate_blackwell_inv_instance(n=40, support_size=3, variant="wb_inv_code", seed=123)
        rng = np.random.default_rng(123)
        bb_res = blackbox_correlation_learner(inst, q=80, rng=rng)
        self.assertGreaterEqual(bb_res["heldout_acc"], 0.85)
        dec = decoy_fooled_baseline(inst)
        self.assertEqual(dec["matches_decoy"], 1.0)
        self.assertLess(dec["heldout_acc"], 0.70)

        # Byte-identical, length-matched refutable control (B8 causal control)
        inst_inv = generate_blackwell_inv_instance(n=40, support_size=3, variant="wb_inv_code", seed=123)
        inst_long = generate_blackwell_inv_instance(n=40, support_size=3, variant="wb_decoy_long_refutable_guard", seed=123)
        code_inv = inst_inv.export_whitebox_code()
        code_long = inst_long.export_whitebox_code()
        self.assertEqual(len(code_inv), len(code_long))
        self.assertIn("_OFF = -180.500000", code_long)
        v_long = validate_blackwell_inv_instance(inst_long, n_random=100, seed=123)
        self.assertTrue(v_long["valid"])

        curve = compute_bb_query_curve(n=40, support_size=3, q_grid=(10, 30, 80), n_trials=60, seed=7)
        self.assertGreater(curve[80]["mean_acc"], curve[10]["mean_acc"])
        self.assertGreater(curve[80]["mean_acc"], 0.93)
        q_eq_low = interpolate_q_eq(0.55, curve)
        q_eq_high = interpolate_q_eq(0.95, curve)
        self.assertLess(q_eq_low, q_eq_high)

    def test_scorer_on_true_decoy_and_invalid_answers(self) -> None:
        for task_type in ("majority3", "dlist3", "dtree2"):
            inst = generate_blackwell_inv_instance(n=40, support_size=3, variant="wb_inv_code", seed=9, task_type=task_type)
            truth = score_rule_prediction(inst, inst.true_support, inst.true_signs)
            self.assertEqual((truth["heldout_acc"], truth["exact_rule"], truth["matches_decoy"]), (1.0, 1.0, 0.0))
            decoy = score_rule_prediction(inst, inst.decoy_support, inst.decoy_signs)
            self.assertEqual(decoy["matches_decoy"], 1.0)
            self.assertLess(decoy["heldout_acc"], 1.0)
            for supp, sgns in (([], []), ([1, 2], [1]), ([0, 1, 40], [1, 1, 1]), ([-1, 2, 3], [1, 1, 1])):
                bad = score_rule_prediction(inst, supp, sgns)
                self.assertEqual(bad["heldout_acc"], 0.5, (task_type, supp))
                self.assertEqual(bad["matches_decoy"], 0.0)

    def test_majority_scoring_ignores_answer_order(self) -> None:
        inst = generate_blackwell_inv_instance(n=40, support_size=3, variant="wb_clean", seed=11)
        order = [2, 0, 1]
        supp = [inst.true_support[i] for i in order]
        sgns = [inst.true_signs[i] for i in order]
        self.assertEqual(score_rule_prediction(inst, supp, sgns)["exact_rule"], 1.0)


if __name__ == "__main__":
    unittest.main()
