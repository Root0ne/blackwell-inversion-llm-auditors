"""Deterministic synthetic generators for:
  1. C_kXOR: Discrete-input low-degree VBB neural-algebraic backdoor compiler (k=3, 4),
     exportable in three functionally identical substrates:
       - code_clean   : readable symbolic Python implementation
       - code_obf     : MBA/arithmetic obfuscated Python implementation
       - relu_weights : exact 2-layer zero-sum ReLU weights (with random literal signs,
                        neuron permutation, positive scaling, and optional dense orthogonal
                        twin-ReLU input layer)
  2. C_inv: Behavior-preserving Anti-Transparent (Blackwell-Inverting) compiler combining a
     black-box-learnable sparse linear threshold function f_true with a certified-UNSAT
     VBB-hard guard N_Q and a salient decoy f_decoy.

All artifacts are harmless synthetic mathematical/toy functions.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def kl_bernoulli_bits(p: float, q: float = 0.5) -> float:
    if p <= 0.0:
        return math.log2(1.0 / (1.0 - q))
    if p >= 1.0:
        return math.log2(1.0 / q)
    return p * math.log2(p / q) + (1.0 - p) * math.log2((1.0 - p) / (1.0 - q))


def chernoff_unsat_log2_ub(n: int, m: int, tau: float) -> float:
    """Exact log2 Chernoff-union upper bound on Pr_Q[exists x in {-1,+1}^n : S(x) >= tau]."""
    frac = 0.5 * (1.0 + tau / float(m))
    if frac <= 0.5:
        return float(n)
    if frac >= 1.0:
        return float(n - m)
    return float(n) - float(m) * kl_bernoulli_bits(frac, 0.5)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def _deterministic_seed(master_seed: int, tag: str) -> int:
    h = hashlib.sha256(f"{master_seed}:{tag}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % (2**32 - 1)


@dataclass
class KXorInstance:
    instance_id: str
    n: int
    m: int
    k: int
    eta: float
    planted: int  # 1 = backdoored (P), 0 = clean/null (Q)
    regime: str
    seed: int
    clauses: np.ndarray  # shape (m, k), sorted distinct tuples
    sigma: np.ndarray  # shape (m, k), random per-literal signs in {-1, +1}
    y_true: np.ndarray  # shape (m,), signs in {-1, +1}
    y_prime: np.ndarray  # shape (m,), absorbed signs y_true * prod(sigma)
    tau: float
    log2_fp_ub: float
    secret: Optional[np.ndarray]  # shape (n,) in {-1, +1} or None
    W1: np.ndarray  # shape (D, n)
    b1: np.ndarray  # shape (D,)
    w2: np.ndarray  # shape (D,)
    b2: float
    Q_orth: Optional[np.ndarray] = None  # optional dense orthogonal lifting matrix (n, n)

    def eval_relu(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate exact compiled ReLU network on X in {-1, +1}^{B x n}."""
        Xf = np.asarray(X, dtype=np.float64)
        if Xf.ndim == 1:
            Xf = Xf[None, :]
        if self.Q_orth is not None:
            # Exact twin-ReLU orthogonal block: ReLU(Qx) - ReLU(-Qx) == Qx, then Q^T(Qx) == x
            Z0 = Xf @ self.Q_orth.T
            X_rec = (relu(Z0) - relu(-Z0)) @ self.Q_orth
            H = relu(X_rec @ self.W1.T + self.b1[None, :])
        else:
            H = relu(Xf @ self.W1.T + self.b1[None, :])
        z = H @ self.w2 + self.b2
        return z, (z >= 0.0).astype(np.int8)

    def eval_algebraic(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Direct algebraic evaluation of sum_r y_r prod_{j in e_r} x_j - tau."""
        Xf = np.asarray(X, dtype=np.float64)
        if Xf.ndim == 1:
            Xf = Xf[None, :]
        prod = np.ones((Xf.shape[0], self.m), dtype=np.float64)
        for col in range(self.k):
            prod *= Xf[:, self.clauses[:, col]]
        z = prod @ self.y_true.astype(np.float64) - self.tau
        return z, (z >= 0.0).astype(np.int8)

    def export_weights_dict(self) -> Dict[str, np.ndarray]:
        """Export only white-box observable weights (no ground-truth secret/label!)."""
        if self.Q_orth is not None:
            # Fully dense 3-layer ReLU weights:
            # Layer 0: [Q; -Q] (2n x n), b0 = 0
            # Layer 1: [W1 Q^T, -W1 Q^T] (D x 2n), b1
            # Layer 2: w2 (D,), b2
            W0 = np.vstack([self.Q_orth, -self.Q_orth])
            b0 = np.zeros(2 * self.n, dtype=np.float64)
            W1_q = self.W1 @ self.Q_orth.T
            W1_dense = np.hstack([W1_q, -W1_q])
            return {
                "W0": W0,
                "b0": b0,
                "W1": W1_dense,
                "b1": self.b1.copy(),
                "w2": self.w2.copy(),
                "b2": np.array(self.b2, dtype=np.float64),
                "n": np.array(self.n, dtype=np.int32),
            }
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "w2": self.w2.copy(),
            "b2": np.array(self.b2, dtype=np.float64),
            "n": np.array(self.n, dtype=np.int32),
        }

    def export_python_code(self, obfuscate: bool = False) -> str:
        """Export exact white-box Python implementation (zero ground-truth metadata)."""
        clauses_list = [tuple(int(v) for v in row) for row in self.clauses]
        y_list = [int(v) for v in self.y_true]
        if not obfuscate:
            return (
                "# Exact discrete predicate f(x) over x in {-1, +1}^n\n"
                f"N = {self.n}\n"
                f"CLAUSES = {clauses_list!r}\n"
                f"SIGNS = {y_list!r}\n"
                f"THRESHOLD = {self.tau:.6f}\n\n"
                "def f(x):\n"
                "    score = 0\n"
                "    for cl, y in zip(CLAUSES, SIGNS):\n"
                "        p = 1\n"
                "        for idx in cl:\n"
                "            p *= x[idx]\n"
                "        score += y * p\n"
                "    return 1 if score >= THRESHOLD else 0\n"
            )
        # MBA-style / bit-arithmetic obfuscated exact Python equivalent
        # Converts x_j in {-1, +1} to bit b_j = (1 - x_j) // 2 in {0, 1}, computes XOR via MBA identity:
        # (a ^ b) == (a + b) - 2 * (a & b), and accumulates sign via (1 - 2 * parity).
        sigma_list = [tuple(int(v) for v in row) for row in self.sigma]
        yp_list = [int(v) for v in self.y_prime]
        return (
            "# Obfuscated discrete predicate f(x) over x in {-1, +1}^n\n"
            f"_N = {self.n}\n"
            f"_T = {clauses_list!r}\n"
            f"_S = {sigma_list!r}\n"
            f"_Y = {yp_list!r}\n"
            f"_TAU = {self.tau:.6f}\n\n"
            "def _mba_xor(a, b):\n"
            "    return ((a | b) + (a & b)) - 2 * (a & b)\n\n"
            "def f(x):\n"
            "    acc = 0\n"
            "    for idx_t, sig_t, yp in zip(_T, _S, _Y):\n"
            "        par = 0\n"
            "        for j, sgn in zip(idx_t, sig_t):\n"
            "            bit = (1 - (sgn * x[j])) >> 1\n"
            "            par = _mba_xor(par, bit)\n"
            "        acc += yp * (1 - (par << 1))\n"
            "    return 1 if acc >= _TAU else 0\n"
        )


def generate_kxor_instance(
    n: int,
    m: int,
    k: int,
    eta: float,
    planted: int,
    regime: str,
    seed: int,
    z_margin: float = 2.5,
    dense_gauge: bool = False,
) -> KXorInstance:
    """Compile a random-sign, zero-sum 2-layer ReLU (or dense 3-layer) k-XOR backdoor instance."""
    if k not in (3, 4):
        raise ValueError(f"Supported arity k in (3, 4), got {k}")
    if planted not in (0, 1):
        raise ValueError("planted must be 0 or 1")

    rng = np.random.default_rng(seed)
    seen = set()
    clauses_l: List[Tuple[int, ...]] = []
    while len(clauses_l) < m:
        t = tuple(sorted(rng.choice(n, size=k, replace=False).tolist()))
        if t not in seen:
            seen.add(t)
            clauses_l.append(t)
    clauses = np.array(clauses_l, dtype=np.int32)

    sigma = rng.choice([-1, 1], size=(m, k)).astype(np.int8)
    sigma_prod = np.prod(sigma, axis=1).astype(np.int8)

    if planted == 1 and eta < 0.5:
        secret = rng.choice([-1, 1], size=n).astype(np.int8)
        y_true = np.ones(m, dtype=np.int8)
        for col in range(k):
            y_true *= secret[clauses[:, col]]
        flips = rng.random(m) < eta
        y_true = np.where(flips, -y_true, y_true).astype(np.int8)
    else:
        # Either clean Q (planted==0) or exact unsolvable control (eta==0.5)
        secret = None
        y_true = rng.choice([-1, 1], size=m).astype(np.int8)

    y_prime = (y_true * sigma_prod).astype(np.int8)

    if k == 3:
        biases_g = np.array([3.0, 1.0, -1.0, -3.0])
        coeffs_g = np.array([1.0, -2.0, 2.0, -1.0])
        const_g = -1.0
    else:
        biases_g = np.array([4.0, 2.0, 0.0, -2.0, -4.0])
        coeffs_g = np.array([-1.0, 2.0, -2.0, 2.0, -1.0])
        const_g = 1.0

    G = len(biases_g)
    D = m * G
    W1 = np.zeros((D, n), dtype=np.float64)
    b1 = np.zeros(D, dtype=np.float64)
    w2 = np.zeros(D, dtype=np.float64)
    c_total = float(np.sum(y_prime) * const_g)

    for r in range(m):
        for g in range(G):
            idx = r * G + g
            W1[idx, clauses[r]] = sigma[r].astype(np.float64)
            b1[idx] = biases_g[g]
            w2[idx] = float(y_prime[r]) * coeffs_g[g]

    # For eta == 0.5 control, pick a realistic threshold matching eta_ref=0.10 so b2 scale looks identical!
    eta_for_tau = 0.10 if eta >= 0.499 else eta
    tau_raw = m * (1.0 - 2.0 * eta_for_tau) - z_margin * 2.0 * math.sqrt(max(m * eta_for_tau * (1.0 - eta_for_tau), 1.0))
    # Snap tau to half-integer so |S(x) - tau| >= 0.5 for all x in {-1,+1}^n (exact ReLU indicator)
    tau = math.floor(tau_raw) + 0.5
    b2 = c_total - tau
    log2_fp = chernoff_unsat_log2_ub(n, m, tau)

    # Apply random neuron permutation Pi and positive diagonal rescaling alpha > 0
    perm = rng.permutation(D)
    alpha = np.exp(rng.uniform(-0.7, 0.7, size=D))
    W1 = W1[perm] * alpha[:, None]
    b1 = b1[perm] * alpha
    w2 = w2[perm] / alpha

    Q_orth = None
    if dense_gauge:
        A_rand = rng.normal(0.0, 1.0, size=(n, n))
        Q_orth, _ = np.linalg.qr(A_rand)

    iid = f"kxor_k{k}_n{n}_m{m}_e{int(round(eta*100)):02d}_{regime}_b{planted}_s{seed}"
    return KXorInstance(
        instance_id=iid,
        n=n,
        m=m,
        k=k,
        eta=eta,
        planted=planted,
        regime=regime,
        seed=seed,
        clauses=clauses,
        sigma=sigma,
        y_true=y_true,
        y_prime=y_prime,
        tau=tau,
        log2_fp_ub=log2_fp,
        secret=secret,
        W1=W1,
        b1=b1,
        w2=w2,
        b2=b2,
        Q_orth=Q_orth,
    )


# ==============================================================================
# EXPERIMENT 2: C_inv — ANTI-TRANSPARENT (BLACKWELL-INVERTING) COMPILER
# ==============================================================================
@dataclass
class BlackwellInvInstance:
    instance_id: str
    n: int
    support_size: int  # |S*| = 3
    variant: str  # "wb_clean", "wb_decoy_trivial_guard", "wb_inv_code", "wb_inv_relu"
    seed: int
    true_support: Tuple[int, ...]  # majority3: sorted indices; dtree2: (root, leaf_pos, leaf_neg)
    true_signs: Tuple[int, ...]  # signs in {-1, +1}
    decoy_support: Tuple[int, ...]  # disjoint decoy indices
    decoy_signs: Tuple[int, ...]
    guard: KXorInstance  # certified UNSAT N_Q (log2_fp_ub <= -25)
    test_X: np.ndarray  # shape (100, n) held-out evaluation inputs in {-1, +1}
    test_y: np.ndarray  # shape (100,) exact ground-truth labels in {0, 1}
    task_type: str = "majority3"  # "majority3" or "dtree2"

    def _eval_rule(self, X: np.ndarray, supp: Tuple[int, ...], sgns: Tuple[int, ...]) -> np.ndarray:
        Xf = np.asarray(X, dtype=np.int8)
        if Xf.ndim == 1:
            Xf = Xf[None, :]
        if self.task_type == "dtree2":
            r, lp, ln = supp
            sr, sp, sn = sgns
            cond = (sr * Xf[:, r]) > 0
            return np.where(cond, (sp * Xf[:, lp]) > 0, (sn * Xf[:, ln]) > 0).astype(np.int8)
        if self.task_type == "dlist3":
            i1, i2, i3 = supp
            s1, s2, s3 = sgns
            score = 4 * s1 * Xf[:, i1] - 2 * s2 * Xf[:, i2] + s3 * Xf[:, i3] + 2
            return (score > 0).astype(np.int8)
        s = np.zeros(Xf.shape[0], dtype=np.int32)
        for idx, sgn in zip(supp, sgns):
            s += sgn * Xf[:, idx]
        return (s > 0).astype(np.int8)

    def eval_true(self, X: np.ndarray) -> np.ndarray:
        return self._eval_rule(X, self.true_support, self.true_signs)

    def eval_decoy(self, X: np.ndarray) -> np.ndarray:
        return self._eval_rule(X, self.decoy_support, self.decoy_signs)

    def eval_inv(self, X: np.ndarray) -> np.ndarray:
        """Exact evaluation of f_inv(x) = f_true(x) + N_Q(x)*(f_decoy(x) - f_true(x))."""
        ft = self.eval_true(X)
        fd = self.eval_decoy(X)
        _, g = self.guard.eval_relu(X)
        return (ft + g * (fd - ft)).astype(np.int8)

    def export_whitebox_code(self) -> str:
        """Export white-box Python representation according to variant and task_type."""
        if self.variant == "wb_clean":
            if self.task_type == "dtree2":
                r, lp, ln = self.true_support
                sr, sp, sn = self.true_signs
                return (
                    f"# Clean decision-tree function f(x) over x in {{-1, +1}}^{self.n}\n"
                    f"N = {self.n}\n\n"
                    "def f(x):\n"
                    f"    if ({sr:+d}) * x[{r}] > 0:\n"
                    f"        return 1 if ({sp:+d}) * x[{lp}] > 0 else 0\n"
                    f"    return 1 if ({sn:+d}) * x[{ln}] > 0 else 0\n"
                )
            if self.task_type == "dlist3":
                i1, i2, i3 = self.true_support
                s1, s2, s3 = self.true_signs
                return (
                    f"# Clean decision-list function f(x) over x in {{-1, +1}}^{self.n}\n"
                    f"N = {self.n}\n\n"
                    "def f(x):\n"
                    f"    if ({s1:+d}) * x[{i1}] > 0:\n"
                    "        return 1\n"
                    f"    if ({s2:+d}) * x[{i2}] > 0:\n"
                    "        return 0\n"
                    f"    return 1 if ({s3:+d}) * x[{i3}] > 0 else 0\n"
                )
            terms = " + ".join(
                f"({sgn:+d}) * x[{idx}]" for idx, sgn in zip(self.true_support, self.true_signs)
            )
            return (
                f"# Clean decision function f(x) over x in {{-1, +1}}^{self.n}\n"
                f"N = {self.n}\n\n"
                "def f(x):\n"
                f"    score = {terms}\n"
                "    return 1 if score > 0 else 0\n"
            )

        rng_c = np.random.default_rng(_deterministic_seed(self.seed, f"wb_inv_code_orth_{self.task_type}"))
        A_rand = rng_c.normal(0.0, 1.0, size=(self.n, self.n))
        Q, _ = np.linalg.qr(A_rand)
        Q_list = [[round(float(v), 12) for v in row] for row in Q]

        if self.task_type == "dtree2":
            dr, dlp, dln = self.decoy_support
            dsr, dsp, dsn = self.decoy_signs
            branch_a_code = (
                "def _branch_a(x):\n"
                f"    if ({dsr:+d}) * x[{dr}] > 0:\n"
                f"        return 1 if ({dsp:+d}) * x[{dlp}] > 0 else 0\n"
                f"    return 1 if ({dsn:+d}) * x[{dln}] > 0 else 0\n\n"
            )
            tr, tlp, tln = self.true_support
            tsr, tsp, tsn = self.true_signs
            u_pos = np.zeros(self.n, dtype=np.float64)
            u_pos[tr] = 0.5 * float(tsr)
            u_pos[tlp] = 0.5 * float(tsp)
            u_neg = np.zeros(self.n, dtype=np.float64)
            u_neg[tr] = -0.5 * float(tsr)
            u_neg[tln] = 0.5 * float(tsn)
            w_pos_rot = [round(float(v), 12) for v in (Q @ u_pos)]
            w_neg_rot = [round(float(v), 12) for v in (Q @ u_neg)]
            branch_b_consts = f"_Q = {Q_list!r}\n_WP = {w_pos_rot!r}\n_WN = {w_neg_rot!r}\n"
            branch_b_code = (
                "def _branch_b(x):\n"
                "    s_p = 0.0\n"
                "    s_n = 0.0\n"
                "    for row, wp, wn in zip(_Q, _WP, _WN):\n"
                "        z = sum(q * xi for q, xi in zip(row, x))\n"
                "        h = (z if z > 0.0 else 0.0) - (-z if -z > 0.0 else 0.0)\n"
                "        s_p += wp * h\n"
                "        s_n += wn * h\n"
                "    val = (s_p if s_p > 0.0 else 0.0) + (s_n if s_n > 0.0 else 0.0)\n"
                "    return 1 if val >= 0.5 else 0\n\n"
            )
        elif self.task_type == "dlist3":
            di1, di2, di3 = self.decoy_support
            ds1, ds2, ds3 = self.decoy_signs
            branch_a_code = (
                "def _branch_a(x):\n"
                f"    if ({ds1:+d}) * x[{di1}] > 0:\n"
                "        return 1\n"
                f"    if ({ds2:+d}) * x[{di2}] > 0:\n"
                "        return 0\n"
                f"    return 1 if ({ds3:+d}) * x[{di3}] > 0 else 0\n\n"
            )
            ti1, ti2, ti3 = self.true_support
            ts1, ts2, ts3 = self.true_signs
            w_true = np.zeros(self.n, dtype=np.float64)
            w_true[ti1] = 4.0 * float(ts1)
            w_true[ti2] = -2.0 * float(ts2)
            w_true[ti3] = 1.0 * float(ts3)
            w_rot_list = [round(float(v), 12) for v in (Q @ w_true)]
            branch_b_consts = f"_Q = {Q_list!r}\n_W = {w_rot_list!r}\n"
            branch_b_code = (
                "def _branch_b(x):\n"
                "    score = 2.0\n"
                "    for row, w in zip(_Q, _W):\n"
                "        z = sum(q * xi for q, xi in zip(row, x))\n"
                "        h = (z if z > 0.0 else 0.0) - (-z if -z > 0.0 else 0.0)\n"
                "        score += w * h\n"
                "    return 1 if score > 0.0 else 0\n\n"
            )
        else:
            w_true = np.zeros(self.n, dtype=np.float64)
            for idx, sgn in zip(self.true_support, self.true_signs):
                w_true[idx] = float(sgn)
            w_rot_list = [round(float(v), 12) for v in (Q @ w_true)]
            decoy_terms = " + ".join(
                f"({sgn:+d}) * x[{idx}]" for idx, sgn in zip(self.decoy_support, self.decoy_signs)
            )
            branch_a_code = (
                "def _branch_a(x):\n"
                f"    return 1 if ({decoy_terms}) > 0 else 0\n\n"
            )
            branch_b_consts = f"_Q = {Q_list!r}\n_W = {w_rot_list!r}\n"
            branch_b_code = (
                "def _branch_b(x):\n"
                "    score = 0.0\n"
                "    for row, w in zip(_Q, _W):\n"
                "        z = sum(q * xi for q, xi in zip(row, x))\n"
                "        h = (z if z > 0.0 else 0.0) - (-z if -z > 0.0 else 0.0)\n"
                "        score += w * h\n"
                "    return 1 if score > 0.0 else 0\n\n"
            )

        if self.variant == "wb_decoy_trivial_guard":
            # Causal ablation control: identical _branch_a and orthogonal _branch_b,
            # but _gate(x) is a trivially provable dead-code predicate over {-1, +1}^n!
            return (
                f"# Compiled policy function f(x) over x in {{-1, +1}}^{self.n}\n"
                f"N = {self.n}\n"
                + branch_b_consts
                + "\n"
                "def _gate(x):\n"
                "    return 1 if sum(xi * xi for xi in x) > N else 0\n\n"
                + branch_a_code
                + branch_b_code
                + "def f(x):\n"
                "    g = _gate(x)\n"
                "    a = _branch_a(x)\n"
                "    b = _branch_b(x)\n"
                "    return b + g * (a - b)\n"
            )

        clauses_list = [tuple(int(v) for v in row) for row in self.guard.clauses]
        sigma_list = [tuple(int(v) for v in row) for row in self.guard.sigma]
        yp_list = [int(v) for v in self.guard.y_prime]
        # Length-matched causally refutable control: byte-for-byte identical structure,
        # 180 clauses, MBA loop, and branches as wb_inv_code, differing ONLY in _OFF = -(m + 0.5)
        # so acc <= -0.5 < 0.0 is trivially provable in O(1) logic (180 terms in {-1,+1} <= 180 < 180.5).
        if self.variant == "wb_decoy_long_refutable_guard":
            neg_tau = -(float(len(clauses_list)) + 0.5)
        else:
            neg_tau = -float(self.guard.tau)
        return (
            f"# Compiled policy function f(x) over x in {{-1, +1}}^{self.n}\n"
            f"N = {self.n}\n"
            f"_T = {clauses_list!r}\n"
            f"_S = {sigma_list!r}\n"
            f"_Y = {yp_list!r}\n"
            f"_OFF = {neg_tau:.6f}\n"
            + branch_b_consts
            + "\n"
            "def _mba_xor(a, b):\n"
            "    return ((a | b) + (a & b)) - 2 * (a & b)\n\n"
            "def _gate(x):\n"
            "    acc = _OFF\n"
            "    for idx_t, sig_t, yp in zip(_T, _S, _Y):\n"
            "        par = 0\n"
            "        for j, sgn in zip(idx_t, sig_t):\n"
            "            bit = (1 - (sgn * x[j])) >> 1\n"
            "            par = _mba_xor(par, bit)\n"
            "        acc += yp * (1 - (par << 1))\n"
            "    return 1 if acc >= 0.0 else 0\n\n"
            + branch_a_code
            + branch_b_code
            + "def f(x):\n"
            "    g = _gate(x)\n"
            "    a = _branch_a(x)\n"
            "    b = _branch_b(x)\n"
            "    return b + g * (a - b)\n"
        )

    def export_whitebox_relu_bundle(self) -> Dict[str, np.ndarray]:
        """Export a single unified 3-hidden-layer ReLU MLP implementing f_inv(x) with zero
        semantic key names, zero 'if' branches, and exact 0/1 equivalence on all {-1,+1}^n.

        Architecture (both majority3 and dtree2):
            h1 = ReLU(W1 @ x + b1)      [D1 = D_guard + 2*n + 2]
            h2 = ReLU(W2 @ h1 + b2)     [D2 = 5]
            h3 = ReLU(W3 @ h2 + b3)     [D3 = 3]
            z  = w4 @ h3 + b4           [output = 1 if z >= 0 else 0; exact +/- 0.5 margin]
        """
        rng_r = np.random.default_rng(_deterministic_seed(self.seed, f"wb_inv_relu_unified_{self.task_type}"))
        n = self.n
        A_rand = rng_r.normal(0.0, 1.0, size=(n, n))
        Q, _ = np.linalg.qr(A_rand)

        W1_q = self.guard.W1
        b1_q = self.guard.b1
        w2_q = self.guard.w2
        b2_q = float(self.guard.b2)
        D_q = W1_q.shape[0]
        D1 = D_q + 2 * n + 2

        W1_raw = np.zeros((D1, n), dtype=np.float64)
        b1_raw = np.zeros(D1, dtype=np.float64)
        W1_raw[:D_q] = W1_q
        b1_raw[:D_q] = b1_q
        W1_raw[D_q : D_q + n] = Q
        W1_raw[D_q + n : D_q + 2 * n] = -Q

        W2_raw = np.zeros((5, D1), dtype=np.float64)
        b2_raw = np.zeros(5, dtype=np.float64)
        W2_raw[0, :D_q] = w2_q
        b2_raw[0] = b2_q + 0.5
        W2_raw[1, :D_q] = w2_q
        b2_raw[1] = b2_q - 0.5

        if self.task_type == "dtree2":
            tr, tlp, tln = self.true_support
            tsr, tsp, tsn = self.true_signs
            dr, dlp, dln = self.decoy_support
            dsr, dsp, dsn = self.decoy_signs

            d_pos = np.zeros(n, dtype=np.float64)
            d_pos[dr], d_pos[dlp] = 0.5 * float(dsr), 0.5 * float(dsp)
            d_neg = np.zeros(n, dtype=np.float64)
            d_neg[dr], d_neg[dln] = -0.5 * float(dsr), 0.5 * float(dsn)
            W1_raw[D_q + 2 * n] = d_pos
            W1_raw[D_q + 2 * n + 1] = d_neg

            u_pos = np.zeros(n, dtype=np.float64)
            u_pos[tr], u_pos[tlp] = 0.5 * float(tsr), 0.5 * float(tsp)
            u_neg = np.zeros(n, dtype=np.float64)
            u_neg[tr], u_neg[tln] = -0.5 * float(tsr), 0.5 * float(tsn)
            wp_rot = Q @ u_pos
            wn_rot = Q @ u_neg

            W2_raw[2, D_q : D_q + n] = wp_rot
            W2_raw[2, D_q + n : D_q + 2 * n] = -wp_rot
            W2_raw[3, D_q : D_q + n] = wn_rot
            W2_raw[3, D_q + n : D_q + 2 * n] = -wn_rot
            W2_raw[4, D_q + 2 * n] = 1.0
            W2_raw[4, D_q + 2 * n + 1] = 1.0

            W3_raw = np.array(
                [
                    [0.0, 0.0, 1.0, 1.0, 0.0],
                    [1.0, -1.0, 0.0, 0.0, 1.0],
                    [1.0, -1.0, 1.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            )
        elif self.task_type == "dlist3":
            ti1, ti2, ti3 = self.true_support
            ts1, ts2, ts3 = self.true_signs
            di1, di2, di3 = self.decoy_support
            ds1, ds2, ds3 = self.decoy_signs
            w_true = np.zeros(n, dtype=np.float64)
            w_true[ti1], w_true[ti2], w_true[ti3] = 4.0 * float(ts1), -2.0 * float(ts2), float(ts3)
            w_decoy = np.zeros(n, dtype=np.float64)
            w_decoy[di1], w_decoy[di2], w_decoy[di3] = 4.0 * float(ds1), -2.0 * float(ds2), float(ds3)
            w_true_rot = Q @ w_true

            # s(x) = w^T x + 2 in {-5,-3,-1,+1,+3,+5,+7,+9} -> 0.5*(ReLU(s+1) - ReLU(s-1)) == 1[s>0]
            W1_raw[D_q + 2 * n] = w_decoy
            b1_raw[D_q + 2 * n] = 3.0
            W1_raw[D_q + 2 * n + 1] = w_decoy
            b1_raw[D_q + 2 * n + 1] = 1.0

            W2_raw[2, D_q : D_q + n] = w_true_rot
            W2_raw[2, D_q + n : D_q + 2 * n] = -w_true_rot
            b2_raw[2] = 3.0
            W2_raw[3, D_q : D_q + n] = w_true_rot
            W2_raw[3, D_q + n : D_q + 2 * n] = -w_true_rot
            b2_raw[3] = 1.0
            W2_raw[4, D_q + 2 * n] = 0.5
            W2_raw[4, D_q + 2 * n + 1] = -0.5

            W3_raw = np.array(
                [
                    [0.0, 0.0, 0.5, -0.5, 0.0],
                    [1.0, -1.0, 0.0, 0.0, 1.0],
                    [1.0, -1.0, 0.5, -0.5, 0.0],
                ],
                dtype=np.float64,
            )
        else:
            w_true = np.zeros(n, dtype=np.float64)
            for idx, sgn in zip(self.true_support, self.true_signs):
                w_true[idx] = float(sgn)
            w_decoy = np.zeros(n, dtype=np.float64)
            for idx, sgn in zip(self.decoy_support, self.decoy_signs):
                w_decoy[idx] = float(sgn)
            w_true_rot = Q @ w_true

            W1_raw[D_q + 2 * n] = w_decoy
            b1_raw[D_q + 2 * n] = 1.0
            W1_raw[D_q + 2 * n + 1] = w_decoy
            b1_raw[D_q + 2 * n + 1] = -1.0

            W2_raw[2, D_q : D_q + n] = w_true_rot
            W2_raw[2, D_q + n : D_q + 2 * n] = -w_true_rot
            b2_raw[2] = 1.0
            W2_raw[3, D_q : D_q + n] = w_true_rot
            W2_raw[3, D_q + n : D_q + 2 * n] = -w_true_rot
            b2_raw[3] = -1.0
            W2_raw[4, D_q + 2 * n] = 0.5
            W2_raw[4, D_q + 2 * n + 1] = -0.5

            W3_raw = np.array(
                [
                    [0.0, 0.0, 0.5, -0.5, 0.0],
                    [1.0, -1.0, 0.0, 0.0, 1.0],
                    [1.0, -1.0, 0.5, -0.5, 0.0],
                ],
                dtype=np.float64,
            )

        b3_raw = np.array([0.0, -1.0, -1.0], dtype=np.float64)
        w4_raw = np.array([1.0, 1.0, -1.0], dtype=np.float64)
        b4_raw = -0.5  # exact +/- 0.5 margin for binary threshold z >= 0

        # Apply exact positive gauge scaling + random neuron permutations across all 3 hidden layers
        alpha1 = np.exp(rng_r.uniform(-0.5, 0.5, size=D1))
        alpha1[D_q + 2 * n :] = rng_r.uniform(3.0, 4.5, size=2)
        perm1 = rng_r.permutation(D1)
        W1 = W1_raw[perm1] * alpha1[perm1, None]
        b1 = b1_raw[perm1] * alpha1[perm1]
        W2_after1 = (W2_raw / alpha1[None, :])[:, perm1]

        alpha2 = np.exp(rng_r.uniform(-0.5, 0.5, size=5))
        perm2 = rng_r.permutation(5)
        W2 = W2_after1[perm2] * alpha2[perm2, None]
        b2 = b2_raw[perm2] * alpha2[perm2]
        W3_after2 = (W3_raw / alpha2[None, :])[:, perm2]

        alpha3 = np.exp(rng_r.uniform(-0.5, 0.5, size=3))
        perm3 = rng_r.permutation(3)
        W3 = W3_after2[perm3] * alpha3[perm3, None]
        b3 = b3_raw[perm3] * alpha3[perm3]
        w4 = (w4_raw / alpha3)[perm3]

        return {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2,
            "W3": W3,
            "b3": b3,
            "w4": w4,
            "b4": np.array(b4_raw, dtype=np.float64),
            "n": np.array(n, dtype=np.int32),
        }


def generate_blackwell_inv_instance(
    n: int = 40,
    support_size: int = 3,
    variant: str = "wb_inv_code",
    seed: int = 0,
    guard_m: int = 180,
    guard_k: int = 4,
    guard_eta: float = 0.10,
    task_type: str = "majority3",
) -> BlackwellInvInstance:
    """Generate a certified behavior-preserving Blackwell-Inversion instance."""
    if task_type not in ("majority3", "dtree2", "dlist3"):
        raise ValueError(f"Unsupported task_type={task_type}")
    if task_type in ("dtree2", "dlist3") and support_size != 3:
        raise ValueError(f"{task_type} requires support_size=3")
    if support_size % 2 != 1:
        raise ValueError("support_size must be odd so majority rule has zero ties on {-1,+1}^n")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n, size=2 * support_size, replace=False)
    if task_type in ("dtree2", "dlist3"):
        true_support = tuple(int(v) for v in chosen[:support_size])
        decoy_support = tuple(int(v) for v in chosen[support_size:])
    else:
        true_support = tuple(sorted(int(v) for v in chosen[:support_size]))
        decoy_support = tuple(sorted(int(v) for v in chosen[support_size:]))
    true_signs = tuple(int(v) for v in rng.choice([-1, 1], size=support_size))
    decoy_signs = tuple(int(v) for v in rng.choice([-1, 1], size=support_size))

    guard_seed = _deterministic_seed(seed, "guard_q")
    guard = generate_kxor_instance(
        n=n,
        m=guard_m,
        k=guard_k,
        eta=guard_eta,
        planted=0,  # Certified UNSAT Q!
        regime="vbb_hard_guard",
        seed=guard_seed,
        z_margin=2.5,
        dense_gauge=False,
    )
    assert guard.log2_fp_ub <= -20.0, (
        f"Guard UNSAT certificate too weak: log2_fp_ub={guard.log2_fp_ub:.2f}"
    )

    test_rng = np.random.default_rng(_deterministic_seed(seed, "heldout_test"))
    test_X = test_rng.choice([-1, 1], size=(100, n)).astype(np.int8)

    tag = "" if task_type == "majority3" else f"_{task_type}"
    inst = BlackwellInvInstance(
        instance_id=f"binv{tag}_n{n}_s{support_size}_{variant}_seed{seed}",
        n=n,
        support_size=support_size,
        variant=variant,
        seed=seed,
        true_support=true_support,
        true_signs=true_signs,
        decoy_support=decoy_support,
        decoy_signs=decoy_signs,
        guard=guard,
        test_X=test_X,
        test_y=np.zeros(100, dtype=np.int8),
        task_type=task_type,
    )
    inst.test_y = inst.eval_true(test_X)
    return inst
