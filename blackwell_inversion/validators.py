"""Independent ground-truth validators for C_kXOR and C_inv instances."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .generators import BlackwellInvInstance, KXorInstance, relu


def eval_exported_weights(bundle: Dict[str, np.ndarray], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate exported white-box weight bundle independently of generator classes.

    Supports:
      - 2-layer ReLU (W1, b1, w2, b2)
      - 3-layer dense-gauge ReLU (W0, b0, W1, b1, w2, b2)
      - 4-layer unified C_inv ReLU MLP (W1, b1, W2, b2, W3, b3, w4, b4)
    """
    Xf = np.asarray(X, dtype=np.float64)
    if Xf.ndim == 1:
        Xf = Xf[None, :]
    if "w4" in bundle:
        H1 = relu(Xf @ bundle["W1"].T + bundle["b1"][None, :])
        H2 = relu(H1 @ bundle["W2"].T + bundle["b2"][None, :])
        H3 = relu(H2 @ bundle["W3"].T + bundle["b3"][None, :])
        z = H3 @ bundle["w4"] + float(bundle["b4"])
    elif "W0" in bundle:
        H0 = relu(Xf @ bundle["W0"].T + bundle["b0"][None, :])
        H1 = relu(H0 @ bundle["W1"].T + bundle["b1"][None, :])
        z = H1 @ bundle["w2"] + float(bundle["b2"])
    else:
        H1 = relu(Xf @ bundle["W1"].T + bundle["b1"][None, :])
        z = H1 @ bundle["w2"] + float(bundle["b2"])
    return z, (z >= 0.0).astype(np.int8)


def eval_exported_python(code: str, X: np.ndarray) -> np.ndarray:
    """Execute generated synthetic Python code in a restricted namespace and score batch X."""
    ns: Dict[str, object] = {"__builtins__": {"zip": zip, "sum": sum, "range": range, "len": len}}
    exec(compile(code, "<synthetic_artifact>", "exec"), ns, ns)
    fn = ns["f"]
    assert callable(fn), "Exported code did not define callable f(x)"
    Xf = np.asarray(X, dtype=np.int64)
    if Xf.ndim == 1:
        Xf = Xf[None, :]
    return np.array([int(fn(row.tolist())) for row in Xf], dtype=np.int8)


def validate_kxor_instance(inst: KXorInstance, n_random: int = 200, seed: int = 999) -> Dict[str, object]:
    """Verify exact equivalence across algebraic, ReLU weights, clean code, and obfuscated code."""
    rng = np.random.default_rng(seed)
    X_test = rng.choice([-1, 1], size=(n_random, inst.n)).astype(np.int8)
    if inst.secret is not None:
        X_test = np.vstack([inst.secret[None, :], (-inst.secret)[None, :], X_test])

    z_alg, y_alg = inst.eval_algebraic(X_test)
    z_rel, y_rel = inst.eval_relu(X_test)
    bundle = inst.export_weights_dict()
    z_exp, y_exp = eval_exported_weights(bundle, X_test)

    max_logit_diff = float(max(np.max(np.abs(z_alg - z_rel)), np.max(np.abs(z_alg - z_exp))))
    assert max_logit_diff < 1e-8, f"ReLU logit mismatch: {max_logit_diff}"
    assert np.array_equal(y_alg, y_rel) and np.array_equal(y_alg, y_exp), "Binary output mismatch"

    y_clean = eval_exported_python(inst.export_python_code(obfuscate=False), X_test[:32])
    y_obf = eval_exported_python(inst.export_python_code(obfuscate=True), X_test[:32])
    assert np.array_equal(y_alg[:32], y_clean), "Clean Python code mismatch"
    assert np.array_equal(y_alg[:32], y_obf), "Obfuscated Python code mismatch"

    # Zero-sum degree-1 weight invariance check on raw 2-layer weights
    deg1_norm = float(np.linalg.norm(inst.W1.T @ inst.w2))
    assert deg1_norm < 1e-10, f"Zero-sum Deg1 property violated: {deg1_norm}"

    trigger_ok = None
    neg_trigger = None
    rand_fps = int(np.sum(y_alg[2:] if inst.secret is not None else y_alg))
    if inst.secret is not None:
        trigger_ok = int(y_alg[0])
        neg_trigger = int(y_alg[1])

    return {
        "valid": True,
        "max_logit_diff": max_logit_diff,
        "deg1_norm": deg1_norm,
        "trigger_fires": trigger_ok,
        "neg_secret_fires": neg_trigger,
        "random_false_triggers": rand_fps,
        "log2_fp_ub": inst.log2_fp_ub,
    }


def validate_blackwell_inv_instance(
    inst: BlackwellInvInstance, n_random: int = 200, seed: int = 1001
) -> Dict[str, object]:
    """Verify exact behavior preservation f_inv(x) == f_true(x) across code and ReLU bundle."""
    rng = np.random.default_rng(seed)
    X_test = np.vstack([inst.test_X, rng.choice([-1, 1], size=(n_random, inst.n)).astype(np.int8)])
    y_true = inst.eval_true(X_test)
    y_inv = inst.eval_inv(X_test)
    assert np.array_equal(y_true, y_inv), "f_inv deviated from f_true!"

    # Verify exported Python code matches f_true identically
    y_code = eval_exported_python(inst.export_whitebox_code(), X_test[:64])
    assert np.array_equal(y_true[:64], y_code), "Exported white-box code deviated from f_true!"

    # Verify unified 3-hidden-layer ReLU MLP matches f_true identically with +/- 0.5 exact margin
    bundle = inst.export_whitebox_relu_bundle()
    z_relu, y_relu_inv = eval_exported_weights(bundle, X_test)
    assert np.array_equal(y_true, y_relu_inv), "Exported unified ReLU MLP deviated from f_true!"
    max_margin_err = float(np.max(np.abs(z_relu - (y_true.astype(np.float64) - 0.5))))
    assert max_margin_err < 1e-8, f"Unified ReLU MLP margin mismatch: {max_margin_err}"

    return {
        "valid": True,
        "discrepancies": 0,
        "n_checked": int(len(X_test)),
        "max_mlp_margin_err": max_margin_err,
        "guard_log2_fp_ub": inst.guard.log2_fp_ub,
        "test_y_balance": float(np.mean(inst.test_y)),
    }
