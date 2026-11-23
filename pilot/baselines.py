"""Reference white-box, black-box, and information-theoretic baselines for Exp 1 (C_kXOR)
and Exp 2 (C_inv).

All evaluation metrics use tie-aware pairwise AUC/Gini and split-half held-out thresholding
to prevent artificial advantage inflation on discrete or tied statistics.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import eigvalsh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import norm

from .generators import (
    BlackwellInvInstance,
    KXorInstance,
    generate_blackwell_inv_instance,
)
from .validators import eval_exported_weights


# ==============================================================================
# TIE-AWARE EVALUATION HELPERS
# ==============================================================================
def pairwise_auc_gini(scores_p: Sequence[float], scores_q: Sequence[float]) -> Tuple[float, float]:
    """Exact tie-aware Mann-Whitney AUC and signed Gini = 2*AUC - 1."""
    sp = np.asarray(scores_p, dtype=np.float64)
    sq = np.asarray(scores_q, dtype=np.float64)
    diff = sp[:, None] - sq[None, :]
    auc = float(np.mean((diff > 0.0) + 0.5 * (diff == 0.0)))
    return auc, 2.0 * auc - 1.0


def cv_adv(scores_p: Sequence[float], scores_q: Sequence[float]) -> Tuple[float, float]:
    """Split-half held-out threshold advantage (Adv = TPR - FPR) and full-sample Gini.

    Half 1 chooses threshold and direction (higher vs lower); Half 2 evaluates strictly out-of-sample.
    """
    sp = np.asarray(scores_p, dtype=np.float64)
    sq = np.asarray(scores_q, dtype=np.float64)
    if len(sp) < 2 or len(sq) < 2:
        _, gini = pairwise_auc_gini(sp, sq)
        return gini, gini
    hp = len(sp) // 2
    hq = len(sq) // 2
    p_tr, p_te = sp[:hp], sp[hp:]
    q_tr, q_te = sq[:hq], sq[hq:]

    _, gini_full = pairwise_auc_gini(sp, sq)
    direction = 1.0 if float(np.mean(p_tr)) >= float(np.mean(q_tr)) else -1.0
    p_tr_d, q_tr_d = direction * p_tr, direction * q_tr
    p_te_d, q_te_d = direction * p_te, direction * q_te

    cands = np.unique(np.concatenate([p_tr_d, q_tr_d]))
    if len(cands) == 1:
        return 0.0, gini_full
    mids = 0.5 * (cands[:-1] + cands[1:])
    best_tr = -2.0
    best_th = float(mids[0])
    for th in mids:
        adv_tr = float(np.mean(p_tr_d >= th) - np.mean(q_tr_d >= th))
        if adv_tr > best_tr:
            best_tr = adv_tr
            best_th = float(th)
    adv_te = float(np.mean(p_te_d >= best_th) - np.mean(q_te_d >= best_th))
    return adv_te, gini_full


# ==============================================================================
# EXPERIMENT 1 (C_kXOR) WHITE-BOX & BLACK-BOX BASELINES
# ==============================================================================
def _effective_W1(bundle: Dict[str, np.ndarray]) -> np.ndarray:
    """Return effective Layer-1 weight matrix (D x n), collapsing optional twin-ReLU W0."""
    W1 = np.asarray(bundle["W1"], dtype=np.float64)
    if "W0" in bundle:
        W0 = np.asarray(bundle["W0"], dtype=np.float64)
        # For W0 = [Q; -Q] and W1_dense = [W1 Q^T, -W1 Q^T], W1_dense @ W0 == 2 * W1
        return 0.5 * (W1 @ W0)
    return W1


def scalar_b2_attack(bundle: Dict[str, np.ndarray]) -> float:
    return float(bundle["b2"])


def deg1_attack(bundle: Dict[str, np.ndarray]) -> float:
    W1 = _effective_W1(bundle)
    w2 = np.asarray(bundle["w2"], dtype=np.float64)
    return float(np.linalg.norm(W1.T @ w2))


def deg2_svd_attack(bundle: Dict[str, np.ndarray]) -> float:
    W1 = _effective_W1(bundle)
    w2 = np.asarray(bundle["w2"], dtype=np.float64)
    M2 = (W1.T * w2[None, :]) @ W1
    return float(np.max(np.abs(eigvalsh(M2))))


def deg2_bias_svd_attack(bundle: Dict[str, np.ndarray]) -> float:
    W1 = _effective_W1(bundle)
    w2 = np.asarray(bundle["w2"], dtype=np.float64)
    b1 = np.asarray(bundle["b1"], dtype=np.float64)
    M2b = (W1.T * (w2 * b1)[None, :]) @ W1
    return float(np.max(np.abs(eigvalsh(M2b))))


def probe_logit_m3_attack(
    bundle: Dict[str, np.ndarray], n_probes: int = 256, seed: int = 12345
) -> float:
    n = int(bundle["n"])
    rng = np.random.default_rng(seed)
    X_probe = rng.choice([-1.0, 1.0], size=(n_probes, n))
    z, _ = eval_exported_weights(bundle, X_probe)
    zc = z - float(np.mean(z))
    return float(np.mean(zc**3))


def canon_extract_clauses(
    bundle: Dict[str, np.ndarray], k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Gauge-invariant Canon-style decompiler: collapses optional dense W0, normalizes W1 rows
    by l_inf norm, groups identical (support, literal-sign) parity gadgets, recovers y'_r from
    the top-bias neuron, and un-absorbs prod(sigma_r) to recover exact (clauses, y_true).
    """
    W1 = _effective_W1(bundle)
    b1 = np.asarray(bundle["b1"], dtype=np.float64)
    w2 = np.asarray(bundle["w2"], dtype=np.float64)
    D, _ = W1.shape

    groups: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], List[Tuple[float, float]]] = {}
    for j in range(D):
        row = W1[j]
        scale = float(np.max(np.abs(row)))
        if scale < 1e-12:
            continue
        supp = tuple(int(v) for v in np.where(np.abs(row) > 0.5 * scale)[0])
        if len(supp) != k:
            continue
        signs = tuple(1 if row[idx] > 0.0 else -1 for idx in supp)
        b_norm = float(b1[j]) / scale
        w2_norm = float(w2[j]) * scale
        groups.setdefault((supp, signs), []).append((b_norm, w2_norm))

    clauses_out: List[Tuple[int, ...]] = []
    y_out: List[int] = []
    # Top-bias neuron (+k) has gadget coefficient +1 for k=3 and -1 for k=4
    top_coeff_sign = 1 if k == 3 else -1
    for (supp, signs), items in groups.items():
        items_sorted = sorted(items, key=lambda t: t[0], reverse=True)
        w_top = items_sorted[0][1]
        y_prime = top_coeff_sign * (1 if w_top > 0.0 else -1)
        sigma_prod = 1
        for s in signs:
            sigma_prod *= s
        clauses_out.append(supp)
        y_out.append(int(y_prime * sigma_prod))

    if not clauses_out:
        return np.zeros((0, k), dtype=np.int32), np.zeros((0,), dtype=np.int8)
    return np.array(clauses_out, dtype=np.int32), np.array(y_out, dtype=np.int8)


def local_short_cover_score(clauses: np.ndarray, y: np.ndarray, k: int) -> float:
    """Short even-cover signed motif statistic:
      - k=3: Z4 (signed 4-clause even covers formed by two pairs with identical size-2 symdiff)
      - k=4: Tri(Z3) (signed 3-clause even covers where e_a XOR e_b == e_c)
    """
    m = len(clauses)
    if m == 0:
        return 0.0
    if k == 3:
        # Pair clauses sharing 2 variables -> size-2 symdiff (u, v)
        pair_map: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for r in range(m):
            a, b, c = int(clauses[r, 0]), int(clauses[r, 1]), int(clauses[r, 2])
            yr = int(y[r])
            pair_map.setdefault((a, b), []).append((c, yr))
            pair_map.setdefault((a, c), []).append((b, yr))
            pair_map.setdefault((b, c), []).append((a, yr))
        edge_sums: Dict[Tuple[int, int], List[int]] = {}
        for lst in pair_map.values():
            L = len(lst)
            for i1 in range(L):
                u, y1 = lst[i1]
                for i2 in range(i1 + 1, L):
                    v, y2 = lst[i2]
                    edge = (u, v) if u < v else (v, u)
                    edge_sums.setdefault(edge, []).append(y1 * y2)
        total = 0.0
        for vals in edge_sums.values():
            if len(vals) >= 2:
                s = float(sum(vals))
                # sum_{i < j} v_i v_j = 0.5 * ((sum v)^2 - sum v^2)
                total += 0.5 * (s * s - float(len(vals)))
        return total

    if k == 4:
        # Check triangles e_r1 XOR e_r2 == e_r3 (requires |e_r1 cap e_r2| == 2)
        clause_lookup: Dict[Tuple[int, ...], int] = {
            tuple(int(v) for v in clauses[r]): int(y[r]) for r in range(m)
        }
        pair_to_clauses: Dict[Tuple[int, int], List[int]] = {}
        for r in range(m):
            cl = clauses[r]
            for i in range(4):
                for j in range(i + 1, 4):
                    pair_to_clauses.setdefault((int(cl[i]), int(cl[j])), []).append(r)
        tri_sum = 0.0
        for r_list in pair_to_clauses.values():
            L = len(r_list)
            for i1 in range(L):
                r1 = r_list[i1]
                s1 = set(int(v) for v in clauses[r1])
                for i2 in range(i1 + 1, L):
                    r2 = r_list[i2]
                    sym = tuple(sorted(s1.symmetric_difference(int(v) for v in clauses[r2])))
                    if len(sym) == 4 and sym in clause_lookup:
                        tri_sum += float(int(y[r1]) * int(y[r2]) * clause_lookup[sym])
        return tri_sum / 3.0

    raise ValueError(f"Unsupported k={k}")


def kikuchi_spectral_score(clauses: np.ndarray, y: np.ndarray, n: int, k: int) -> float:
    """Gauge-invariant spectral Kikuchi distinguisher:
      - k=3: Degree-normalized Kikuchi-1 (singleton pair-cancellation adjacency matrix)
      - k=4: Degree-regularized Kikuchi-2 (pair-indexed bipartite cancellation matrix)
    """
    m = len(clauses)
    if m == 0:
        return 0.0
    if k == 3:
        pair_map: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
        for r in range(m):
            a, b, c = int(clauses[r, 0]), int(clauses[r, 1]), int(clauses[r, 2])
            yr = float(y[r])
            pair_map.setdefault((a, b), []).append((c, yr))
            pair_map.setdefault((a, c), []).append((b, yr))
            pair_map.setdefault((b, c), []).append((a, yr))
        S = np.zeros((n, n), dtype=np.float64)
        C_cnt = np.zeros((n, n), dtype=np.float64)
        for lst in pair_map.values():
            L = len(lst)
            if L >= 2:
                for i1 in range(L):
                    u, y1 = lst[i1]
                    for i2 in range(i1 + 1, L):
                        v, y2 = lst[i2]
                        prod = y1 * y2
                        S[u, v] += prod
                        S[v, u] += prod
                        C_cnt[u, v] += 1.0
                        C_cnt[v, u] += 1.0
        deg = np.sum(C_cnt, axis=1)
        inv_sq = 1.0 / np.sqrt(np.maximum(deg, 1.0))
        S_norm = (S * inv_sq[:, None]) * inv_sq[None, :]
        return float(eigvalsh(S_norm)[-1])

    if k == 4:
        N_pairs = n * (n - 1) // 2

        def pid(u: int, v: int) -> int:
            return u * (2 * n - u - 1) // 2 + (v - u - 1)

        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        for r in range(m):
            a, b, c, d = (int(clauses[r, i]) for i in range(4))
            val = float(y[r])
            for p1, p2 in [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]:
                i, j = pid(*p1), pid(*p2)
                rows.extend([i, j])
                cols.extend([j, i])
                vals.extend([val, val])
        M = coo_matrix((vals, (rows, cols)), shape=(N_pairs, N_pairs)).tocsr()
        deg = np.array(np.abs(M).sum(axis=1)).ravel()
        dbar = max(float(np.mean(deg)), 1.0)
        inv_sq = 1.0 / np.sqrt(deg + dbar)
        Mreg = M.multiply(inv_sq[:, None]).multiply(inv_sq[None, :]).tocsr()
        if Mreg.nnz == 0:
            return 0.0
        ev = eigsh(Mreg, k=1, which="LA", return_eigenvectors=False, maxiter=2000)
        return float(ev[0])

    raise ValueError(f"Unsupported k={k}")


def it_exact_lrt(clauses: np.ndarray, y: np.ndarray, n: int, eta: float) -> float:
    """Exact log-likelihood ratio log(dP/dQ) by exhaustive enumeration (only for small n <= 14)."""
    if n > 14:
        raise ValueError(f"it_exact_lrt only supports n <= 14, got n={n}")
    if eta >= 0.5 or eta <= 0.0:
        return 0.0
    num_states = 1 << n
    idx_grid = np.arange(num_states, dtype=np.int32)
    X = np.empty((num_states, n), dtype=np.int8)
    for j in range(n):
        X[:, j] = np.where((idx_grid >> j) & 1, 1, -1).astype(np.int8)
    prod = np.ones((num_states, len(clauses)), dtype=np.int8)
    for col in range(clauses.shape[1]):
        prod *= X[:, clauses[:, col]]
    m_sat = np.sum(prod == y[None, :], axis=1).astype(np.float64)
    m = float(len(clauses))
    log_terms = m_sat * math.log(2.0 * (1.0 - eta)) + (m - m_sat) * math.log(2.0 * eta)
    max_lt = float(np.max(log_terms))
    log_avg = max_lt + math.log(float(np.mean(np.exp(log_terms - max_lt))))
    return float(log_avg)


def it_chernoff_certificate(inst: KXorInstance) -> Dict[str, float]:
    """Analytical IT distinguishability certificate using Chernoff union bound on Q and
    planted tail bound on P (when eta < 0.5).
    """
    fp_ub = min(1.0, 2.0 ** inst.log2_fp_ub) if inst.log2_fp_ub < 0.0 else 1.0
    if inst.eta >= 0.5:
        return {"log2_fp_ub": inst.log2_fp_ub, "fp_ub": fp_ub, "it_adv_lb": 0.0}
    mean_p = inst.m * (1.0 - 2.0 * inst.eta)
    std_p = 2.0 * math.sqrt(max(inst.m * inst.eta * (1.0 - inst.eta), 1e-9))
    z_val = (mean_p - inst.tau) / std_p
    fn_approx = float(norm.cdf(-z_val))
    adv_lb = max(0.0, 1.0 - fn_approx - fp_ub)
    return {
        "log2_fp_ub": float(inst.log2_fp_ub),
        "fp_ub": float(fp_ub),
        "fn_approx": fn_approx,
        "it_adv_lb": float(adv_lb),
    }


def _krawtchouk_norm(n: int, k: int, d: int) -> float:
    """Exact hypergeometric expectation E[prod_{j in e} (x_j s*_j)] at Hamming distance d."""
    denom = math.comb(n, k)
    num = 0
    for t in range(min(k, d) + 1):
        if k - t <= n - d:
            num += ((-1) ** t) * math.comb(d, t) * math.comb(n - d, k - t)
    return float(num) / float(denom)


def _kl_bernoulli(a: float, p: float) -> float:
    if a <= p:
        return 0.0
    if a >= 1.0:
        return -math.log(p) if p > 0.0 else float("inf")
    return a * math.log(a / p) + (1.0 - a) * math.log((1.0 - a) / (1.0 - p))


def blackbox_adv_upper_bound(
    n: int,
    k: int,
    q: int,
    m: Optional[int] = None,
    eta: Optional[float] = None,
    z_alpha: float = 2.5,
) -> float:
    """Information-theoretic upper bound on adaptive q-query black-box advantage against C_kXOR.

    When (m, eta) are supplied, computes the exact Hoeffding-Chernoff Hamming-ball bound:
        Adv_BB(q) <= 2^{-lambda} + q * E[|T_P|] / 2^n,
    accounting for both clean-network false triggers (N_Q) and the full Hamming neighborhood
    T_P = {x : N_P(x) = 1} around the planted secret (and -s* for even k).
    When (m, eta) are omitted, returns the point-function baseline (1 or 2 triggers)/2^n.
    """
    if m is None or eta is None:
        n_triggers = 1 if (k % 2 == 1) else 2
        return min(1.0, float(n_triggers * q) / float(2**n))

    tau_raw = m * (1.0 - 2.0 * eta) - z_alpha * 2.0 * math.sqrt(max(m * eta * (1.0 - eta), 1e-12))
    tau = math.floor(tau_raw) + 0.5
    a = 0.5 * (1.0 + tau / float(m))
    if a >= 1.0:
        return 0.0
    # Clean-network union bound Pr[exists x: N_Q(x)=1] <= 2^{log2_fp_ub}
    log2_fp_ub = float(n) - float(m) * (_kl_bernoulli(a, 0.5) / math.log(2.0))
    fp_ub = 1.0 if log2_fp_ub >= 0.0 else (2.0 ** log2_fp_ub if log2_fp_ub > -1000.0 else 0.0)

    # Expected trigger fraction E[|T_P|] / 2^n via Hoeffding (1963) + shell-wise Chernoff
    ln2 = math.log(2.0)
    frac_ub = 0.0
    for d in range(n + 1):
        mu_d = _krawtchouk_norm(n, k, d)
        p_d = 0.5 * (1.0 + (1.0 - 2.0 * eta) * mu_d)
        kl_val = _kl_bernoulli(a, p_d)
        shell_log2 = (math.lgamma(n + 1) - math.lgamma(d + 1) - math.lgamma(n - d + 1)) / ln2
        p_log2 = -float(m) * kl_val / ln2 if kl_val > 0.0 else 0.0
        term_log2 = shell_log2 + p_log2 - float(n)
        if term_log2 > -1000.0:
            frac_ub += 2.0 ** term_log2
    return min(1.0, fp_ub + float(q) * frac_ub)


def blackbox_random_queries(inst: KXorInstance, q: int, rng: np.random.Generator) -> int:
    """Simulate q uniform black-box queries; predicts PLANTED (1) iff any query returns 1."""
    if q <= 0:
        return 0
    X = rng.choice([-1, 1], size=(q, inst.n)).astype(np.int8)
    _, y = inst.eval_algebraic(X)
    return 1 if int(np.any(y == 1)) else 0


def simulated_annealing_attack(
    clauses: np.ndarray,
    y: np.ndarray,
    n: int,
    restarts: int = 4,
    steps: int = 12000,
    seed: int = 0,
) -> float:
    """Heuristic spin-glass local search (Simulated Annealing) baseline over {-1, +1}^n."""
    m, k = clauses.shape
    if m == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    adj_lists: List[List[int]] = [[] for _ in range(n)]
    for r in range(m):
        for col in range(k):
            adj_lists[int(clauses[r, col])].append(r)
    adj = [np.array(lst, dtype=np.int32) for lst in adj_lists]

    best_all = -float(m)
    T0, T1 = 4.0, 0.3
    ratio = T1 / T0
    for _ in range(restarts):
        x = rng.choice([-1, 1], size=n).astype(np.int8)
        terms = y.astype(np.int32).copy()
        for col in range(k):
            terms *= x[clauses[:, col]]
        score = int(np.sum(terms))
        if score > best_all:
            best_all = float(score)
        for st in range(steps):
            v = int(rng.integers(0, n))
            idx = adj[v]
            if len(idx) == 0:
                continue
            delta = -2 * int(np.sum(terms[idx]))
            if delta >= 0:
                accept = True
            else:
                T = T0 * (ratio ** (st / steps))
                accept = rng.random() < math.exp(delta / T)
            if accept:
                x[v] = -x[v]
                terms[idx] = -terms[idx]
                score += delta
                if score > best_all:
                    best_all = float(score)
    return float(best_all)


def run_kxor_attack_battery(
    instances: Sequence[KXorInstance], seed: int = 12345, include_sa: bool = False
) -> Dict[str, Dict[str, float]]:
    """Run all white-box reference attacks on a balanced set of KXorInstance objects."""
    names = [
        "scalar_b2",
        "deg1",
        "deg2_svd",
        "deg2_bias_svd",
        "probe_m3",
        "canon_short_cover",
        "canon_kikuchi",
    ]
    if include_sa:
        names.append("canon_sa")
    scores_p: Dict[str, List[float]] = {nm: [] for nm in names}
    scores_q: Dict[str, List[float]] = {nm: [] for nm in names}

    for idx, inst in enumerate(instances):
        bundle = inst.export_weights_dict()
        cl, ys = canon_extract_clauses(bundle, inst.k)
        vals = {
            "scalar_b2": scalar_b2_attack(bundle),
            "deg1": deg1_attack(bundle),
            "deg2_svd": deg2_svd_attack(bundle),
            "deg2_bias_svd": deg2_bias_svd_attack(bundle),
            "probe_m3": probe_logit_m3_attack(bundle, n_probes=128, seed=seed + idx),
            "canon_short_cover": local_short_cover_score(cl, ys, inst.k),
            "canon_kikuchi": kikuchi_spectral_score(cl, ys, inst.n, inst.k),
        }
        if include_sa:
            vals["canon_sa"] = simulated_annealing_attack(cl, ys, inst.n, seed=seed + idx)
        target = scores_p if inst.planted == 1 else scores_q
        for nm, v in vals.items():
            target[nm].append(v)

    summary: Dict[str, Dict[str, float]] = {}
    for nm in names:
        adv, gini = cv_adv(scores_p[nm], scores_q[nm])
        summary[nm] = {
            "cv_adv": adv,
            "gini": gini,
            "mean_P": float(np.mean(scores_p[nm])) if scores_p[nm] else 0.0,
            "mean_Q": float(np.mean(scores_q[nm])) if scores_q[nm] else 0.0,
        }
    return summary


def vectorized_sa_attack(
    clauses: np.ndarray,
    y: np.ndarray,
    n: int,
    n_chains: int = 128,
    sweeps: int = 200,
    seed: int = 0,
) -> float:
    """High-throughput vectorized multi-chain NumPy Simulated Annealing attacker."""
    m, k = clauses.shape
    if m == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    X = rng.choice([-1, 1], size=(n_chains, n)).astype(np.int8)
    adj_lists: List[List[int]] = [[] for _ in range(n)]
    for r in range(m):
        for col in range(k):
            adj_lists[int(clauses[r, col])].append(r)
    adj = [np.array(lst, dtype=np.int32) for lst in adj_lists]

    T = np.broadcast_to(y[None, :], (n_chains, m)).astype(np.int16).copy()
    for col in range(k):
        T *= X[:, clauses[:, col]]
    sc = np.sum(T, axis=1)
    best = int(np.max(sc))
    for sw in range(sweeps):
        temp = 3.5 * ((0.15 / 3.5) ** (sw / max(sweeps - 1, 1)))
        for v in rng.permutation(n):
            idx = adj[int(v)]
            if len(idx) == 0:
                continue
            delta = -2 * np.sum(T[:, idx], axis=1)
            accept = (delta >= 0) | (rng.random(n_chains) < np.exp(np.clip(delta / temp, -50.0, 0.0)))
            if np.any(accept):
                X[accept, v] = -X[accept, v]
                T[np.ix_(accept, idx)] = -T[np.ix_(accept, idx)]
                sc[accept] += delta[accept]
        cur = int(np.max(sc))
        if cur > best:
            best = cur
    return float(best)


# ==============================================================================
# EXPERIMENT 2 (C_inv) BLACK-BOX & DECOY BASELINES
# ==============================================================================
def _dtree2_equiv(
    s1: Sequence[int], sg1: Sequence[int], s2: Sequence[int], sg2: Sequence[int]
) -> bool:
    if len(s1) != 3 or len(s2) != 3:
        return False
    a = (int(s1[0]), int(s1[1]), int(s1[2]), int(sg1[0]), int(sg1[1]), int(sg1[2]))
    b1 = (int(s2[0]), int(s2[1]), int(s2[2]), int(sg2[0]), int(sg2[1]), int(sg2[2]))
    b2 = (int(s2[0]), int(s2[2]), int(s2[1]), -int(sg2[0]), int(sg2[2]), int(sg2[1]))
    return a == b1 or a == b2


def score_rule_prediction(
    inst: BlackwellInvInstance,
    pred_support: Sequence[int],
    pred_signs: Sequence[int],
) -> Dict[str, float]:
    """Evaluate a predicted majority3 or dtree2 rule (support, signs) on inst.test_X."""
    supp = [int(i) for i in pred_support]
    sgns = [1 if int(s) >= 0 else -1 for s in pred_signs]
    if len(supp) != len(sgns) or len(supp) == 0 or any(i < 0 or i >= inst.n for i in supp):
        return {
            "heldout_acc": 0.5,
            "support_f1": 0.0,
            "exact_support": 0.0,
            "exact_rule": 0.0,
            "matches_decoy": 0.0,
        }
    if inst.task_type == "dtree2" and len(supp) == 3:
        r, lp, ln = supp
        sr, sp, sn = sgns
        y_hat = np.where((sr * inst.test_X[:, r]) > 0, (sp * inst.test_X[:, lp]) > 0, (sn * inst.test_X[:, ln]) > 0).astype(np.int8)
    elif inst.task_type == "dlist3" and len(supp) == 3:
        i1, i2, i3 = supp
        s1, s2, s3 = sgns
        sc = 4 * s1 * inst.test_X[:, i1] - 2 * s2 * inst.test_X[:, i2] + s3 * inst.test_X[:, i3] + 2
        y_hat = (sc > 0).astype(np.int8)
    else:
        s = np.zeros(inst.test_X.shape[0], dtype=np.int32)
        for idx, sg in zip(supp, sgns):
            s += sg * inst.test_X[:, idx]
        y_hat = (s > 0).astype(np.int8)
    acc = float(np.mean(y_hat == inst.test_y))

    true_set = set(inst.true_support)
    pred_set = set(supp)
    inter = len(true_set & pred_set)
    f1 = (2.0 * inter) / float(len(true_set) + len(pred_set))
    exact_supp = 1.0 if pred_set == true_set else 0.0

    if inst.task_type == "dtree2":
        exact_rule = 1.0 if _dtree2_equiv(supp, sgns, inst.true_support, inst.true_signs) else 0.0
        matches_decoy = 1.0 if _dtree2_equiv(supp, sgns, inst.decoy_support, inst.decoy_signs) else 0.0
    elif inst.task_type == "dlist3":
        exact_rule = 1.0 if (tuple(supp) == inst.true_support and tuple(sgns) == inst.true_signs) else 0.0
        matches_decoy = 1.0 if (tuple(supp) == inst.decoy_support and tuple(sgns) == inst.decoy_signs) else 0.0
    else:
        true_map = dict(zip(inst.true_support, inst.true_signs))
        pred_map = dict(zip(supp, sgns))
        exact_rule = 1.0 if pred_map == true_map else 0.0
        decoy_map = dict(zip(inst.decoy_support, inst.decoy_signs))
        matches_decoy = 1.0 if pred_map == decoy_map else 0.0

    return {
        "heldout_acc": acc,
        "support_f1": f1,
        "exact_support": exact_supp,
        "exact_rule": exact_rule,
        "matches_decoy": matches_decoy,
    }


def blackbox_correlation_learner(
    inst: BlackwellInvInstance, q: int, rng: np.random.Generator
) -> Dict[str, object]:
    """Optimal-order single-coordinate correlation learner for sparse majority over {-1,+1}^n."""
    if q <= 0:
        return {
            "pred_support": [],
            "pred_signs": [],
            "heldout_acc": 0.5,
            "support_f1": 0.0,
            "exact_support": 0.0,
            "exact_rule": 0.0,
            "matches_decoy": 0.0,
        }
    X = rng.choice([-1, 1], size=(q, inst.n)).astype(np.int8)
    y = inst.eval_inv(X)
    y_pm = (2 * y.astype(np.float64)) - 1.0
    corrs = (X.astype(np.float64).T @ y_pm) / float(q)
    top_idx = np.argsort(-np.abs(corrs))[: inst.support_size]
    top_idx_sorted = sorted(int(i) for i in top_idx)
    pred_signs = [1 if corrs[i] >= 0.0 else -1 for i in top_idx_sorted]
    metrics = score_rule_prediction(inst, top_idx_sorted, pred_signs)
    return {"pred_support": top_idx_sorted, "pred_signs": pred_signs, **metrics}


def decoy_fooled_baseline(inst: BlackwellInvInstance) -> Dict[str, float]:
    """Utility when an observer trusts the salient decoy branch f_decoy."""
    return score_rule_prediction(inst, inst.decoy_support, inst.decoy_signs)


def compute_bb_query_curve(
    n: int = 40,
    support_size: int = 3,
    q_grid: Sequence[int] = (5, 10, 15, 20, 30, 40, 60, 80, 120),
    n_trials: int = 200,
    seed: int = 42,
    task_type: str = "majority3",
) -> Dict[int, Dict[str, float]]:
    """Calibrate empirical black-box learning curve u_BB(q) for computing q_eq and Gamma_WB."""
    rng = np.random.default_rng(seed)
    curve: Dict[int, Dict[str, float]] = {}
    for q in q_grid:
        accs = []
        exacts = []
        for _ in range(n_trials):
            supp = rng.choice(n, size=support_size, replace=False)
            sgns = rng.choice([-1, 1], size=support_size)
            X_tr = rng.choice([-1, 1], size=(q, n)).astype(np.int8)
            X_te = rng.choice([-1, 1], size=(100, n)).astype(np.int8)

            if task_type == "dlist3" and support_size == 3:
                i1, i2, i3 = int(supp[0]), int(supp[1]), int(supp[2])
                s1, s2, s3 = int(sgns[0]), int(sgns[1]), int(sgns[2])
                y_tr_bin = ((4 * s1 * X_tr[:, i1] - 2 * s2 * X_tr[:, i2] + s3 * X_tr[:, i3] + 2) > 0).astype(np.int8)
                y_te = ((4 * s1 * X_te[:, i1] - 2 * s2 * X_te[:, i2] + s3 * X_te[:, i3] + 2) > 0).astype(np.int8)
                # Sequential greedy correlation decision-list learner
                y_pm = y_tr_bin.astype(np.float64) * 2.0 - 1.0
                c1 = (X_tr.astype(np.float64).T @ y_pm) / float(q)
                p1 = int(np.argmax(np.abs(c1)))
                ps1 = 1 if c1[p1] >= 0.0 else -1
                m2 = (ps1 * X_tr[:, p1]) < 0
                if int(np.sum(m2)) >= 1:
                    c2 = (X_tr[m2].astype(np.float64).T @ y_pm[m2]) / float(np.sum(m2))
                    c2[p1] = 0.0
                    p2 = int(np.argmax(np.abs(c2)))
                    ps2 = 1 if (-c2[p2]) >= 0.0 else -1
                else:
                    p2 = (p1 + 1) % n
                    ps2 = 1
                m3 = m2 & ((ps2 * X_tr[:, p2]) < 0)
                if int(np.sum(m3)) >= 1:
                    c3 = (X_tr[m3].astype(np.float64).T @ y_pm[m3]) / float(np.sum(m3))
                    c3[p1] = 0.0
                    c3[p2] = 0.0
                    p3 = int(np.argmax(np.abs(c3)))
                    ps3 = 1 if c3[p3] >= 0.0 else -1
                else:
                    rem = [idx for idx in range(n) if idx not in (p1, p2)]
                    p3 = rem[0]
                    ps3 = 1
                y_hat = ((4 * ps1 * X_te[:, p1] - 2 * ps2 * X_te[:, p2] + ps3 * X_te[:, p3] + 2) > 0).astype(np.int8)
                accs.append(float(np.mean(y_hat == y_te)))
                exacts.append(1.0 if ((p1, p2, p3) == (i1, i2, i3) and (ps1, ps2, ps3) == (s1, s2, s3)) else 0.0)
            else:
                y_tr = ((X_tr[:, supp] @ sgns) > 0).astype(np.float64) * 2.0 - 1.0
                corrs = (X_tr.astype(np.float64).T @ y_tr) / float(q)
                top = np.argsort(-np.abs(corrs))[:support_size]
                pred_sgns = np.where(corrs[top] >= 0.0, 1, -1)

                y_te = ((X_te[:, supp] @ sgns) > 0).astype(np.int8)
                y_hat = ((X_te[:, top] @ pred_sgns) > 0).astype(np.int8)
                accs.append(float(np.mean(y_hat == y_te)))

                true_map = dict(zip(supp.tolist(), sgns.tolist()))
                pred_map = dict(zip(top.tolist(), pred_sgns.tolist()))
                exacts.append(1.0 if pred_map == true_map else 0.0)
        curve[int(q)] = {
            "mean_acc": float(np.mean(accs)),
            "se_acc": float(np.std(accs, ddof=1) / math.sqrt(len(accs))),
            "mean_exact_rule": float(np.mean(exacts)),
        }
    return curve


def interpolate_q_eq(
    u_wb: float, bb_curve: Dict[int, Dict[str, float]], metric_key: str = "mean_acc"
) -> float:
    """Invert monotone empirical u_BB(q) curve to compute equivalent black-box queries q_eq."""
    qs = sorted(bb_curve.keys())
    # Enforce monotonic envelope (PAVA-like cumulative max)
    us: List[float] = []
    running = 0.5
    for q in qs:
        running = max(running, float(bb_curve[q][metric_key]))
        us.append(running)

    if u_wb <= 0.5:
        return 0.0
    if u_wb <= us[0]:
        # Linear interpolation from (0, 0.5) to (qs[0], us[0])
        denom = max(us[0] - 0.5, 1e-9)
        return float(qs[0]) * ((u_wb - 0.5) / denom)
    for i in range(len(qs) - 1):
        if us[i] <= u_wb <= us[i + 1]:
            denom = max(us[i + 1] - us[i], 1e-9)
            frac = (u_wb - us[i]) / denom
            return float(qs[i] + frac * (qs[i + 1] - qs[i]))
    return float(qs[-1])
