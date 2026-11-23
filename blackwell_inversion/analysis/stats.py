"""Statistical primitives shared by all analysis scripts.

  - exact Fisher test with a Newcombe-Wilson 95% CI for the distinguishing advantage
    Adv = TPR - FPR (Experiment 1)
  - paired sign-flip permutation test with a paired bootstrap 95% CI for Delta_WB = WB - BB,
    and an exact McNemar test on paired binary outcomes (Experiment 2)
  - Holm-Bonferroni step-down adjustment and PAVA isotonic regression
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest, fisher_exact, norm


# ==============================================================================
# CORE STATISTICAL PRIMITIVES
# ==============================================================================
def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion k/n."""
    if n <= 0:
        return 0.0, 1.0
    z = float(norm.ppf(1.0 - alpha / 2.0))
    p = float(k) / float(n)
    denom = 1.0 + (z * z) / float(n)
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p)) / float(n) + (z * z) / (4.0 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def newcombe_wilson_diff_ci(
    k1: int, n1: int, k0: int, n0: int, alpha: float = 0.05
) -> Tuple[float, float, float]:
    """Newcombe-Wilson (Method 10) CI for difference of two independent proportions p1 - p0."""
    if n1 <= 0 or n0 <= 0:
        return 0.0, -1.0, 1.0
    p1 = float(k1) / float(n1)
    p0 = float(k0) / float(n0)
    l1, u1 = wilson_ci(k1, n1, alpha=alpha)
    l0, u0 = wilson_ci(k0, n0, alpha=alpha)
    diff = p1 - p0
    lo = diff - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    hi = diff + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return diff, max(-1.0, lo), min(1.0, hi)


def fisher_adv_test(
    tp: int, n_pos: int, fp: int, n_neg: int, alternative: str = "greater"
) -> Dict[str, float]:
    """Exact Fisher test + Newcombe-Wilson 95% CI for distinguishing advantage Adv = TPR - FPR."""
    fn = max(0, n_pos - tp)
    tn = max(0, n_neg - fp)
    _, p_val = fisher_exact([[tp, fn], [fp, tn]], alternative=alternative)
    adv, ci_lo, ci_hi = newcombe_wilson_diff_ci(tp, n_pos, fp, n_neg, alpha=0.05)
    return {
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "tpr": float(tp) / float(n_pos) if n_pos > 0 else 0.0,
        "fpr": float(fp) / float(n_neg) if n_neg > 0 else 0.0,
        "adv": float(adv),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_value": float(p_val),
    }


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Step-down Holm-Bonferroni FWER-adjusted p-values (preserves input order)."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(float(p) for p in p_values), key=lambda t: t[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, p * float(m - rank))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return adjusted


def pava_isotonic(y: Sequence[float], weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """Pool Adjacent Violators Algorithm (PAVA) for non-decreasing isotonic regression."""
    vals = [float(v) for v in y]
    n = len(vals)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    wts = [1.0] * n if weights is None else [float(w) for w in weights]
    # Blocks stored as [weighted_sum, total_weight, start_idx, end_idx]
    blocks: List[List[float]] = []
    for i in range(n):
        blocks.append([vals[i] * wts[i], wts[i], float(i), float(i)])
        while len(blocks) >= 2:
            mean_prev = blocks[-2][0] / blocks[-2][1]
            mean_curr = blocks[-1][0] / blocks[-1][1]
            if mean_prev > mean_curr:
                blocks[-2][0] += blocks[-1][0]
                blocks[-2][1] += blocks[-1][1]
                blocks[-2][3] = blocks[-1][3]
                blocks.pop()
            else:
                break
    out = np.zeros(n, dtype=np.float64)
    for wsum, wtot, st, en in blocks:
        mean_val = wsum / wtot
        for idx in range(int(st), int(en) + 1):
            out[idx] = mean_val
    return out


def paired_delta_stats(
    u_wb: Sequence[float],
    u_bb: Sequence[float],
    *,
    alternative: str = "less",
    n_perm: int = 5000,
    n_boot: int = 5000,
    seed: int = 2026,
) -> Dict[str, float]:
    """Exact/Monte-Carlo paired sign-flip permutation test and bootstrap 95% CI for Delta_WB = WB - BB."""
    wb = np.asarray(u_wb, dtype=np.float64)
    bb = np.asarray(u_bb, dtype=np.float64)
    if len(wb) != len(bb) or len(wb) == 0:
        return {
            "n_pairs": 0,
            "mean_bb": 0.0,
            "mean_wb": 0.0,
            "delta_wb": 0.0,
            "ci_lo": 0.0,
            "ci_hi": 0.0,
            "p_value": 1.0,
        }
    diffs = wb - bb
    n = len(diffs)
    obs = float(np.mean(diffs))
    rng = np.random.default_rng(seed)

    # Bootstrap 95% CI
    idx_b = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(diffs[idx_b], axis=1)
    ci_lo = float(np.quantile(boot_means, 0.025))
    ci_hi = float(np.quantile(boot_means, 0.975))

    # Sign-flip permutation test
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = np.mean(signs * diffs[None, :], axis=1)
    if alternative == "less":
        p_val = float((1 + np.sum(perm_means <= obs + 1e-12)) / (n_perm + 1))
    elif alternative == "greater":
        p_val = float((1 + np.sum(perm_means >= obs - 1e-12)) / (n_perm + 1))
    else:
        p_val = float((1 + np.sum(np.abs(perm_means) >= abs(obs) - 1e-12)) / (n_perm + 1))

    return {
        "n_pairs": int(n),
        "mean_bb": float(np.mean(bb)),
        "mean_wb": float(np.mean(wb)),
        "delta_wb": obs,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_val,
    }


def mcnemar_exact(wb_ok: Sequence[int], bb_ok: Sequence[int], alternative: str = "less") -> Dict[str, Any]:
    """Exact binomial McNemar test on paired binary outcomes (WB vs BB)."""
    b01 = 0  # BB=1, WB=0 (WB worse)
    b10 = 0  # BB=0, WB=1 (WB better)
    for w, b in zip(wb_ok, bb_ok):
        if int(w) == 1 and int(b) == 0:
            b10 += 1
        elif int(w) == 0 and int(b) == 1:
            b01 += 1
    discordant = b10 + b01
    if discordant == 0:
        return {"wb_better": b10, "wb_worse": b01, "p_value": 1.0}
    res = binomtest(b10, n=discordant, p=0.5, alternative=alternative)
    return {"wb_better": b10, "wb_worse": b01, "p_value": float(res.pvalue)}


def load_jsonl(path: Path, missing_ok: bool = False) -> List[Dict[str, Any]]:
    """All records of a JSONL log; blank lines are skipped. A missing file is an error unless missing_ok."""
    if missing_ok and not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
