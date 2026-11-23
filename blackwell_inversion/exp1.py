"""Experiment 1 (`C_kXOR`): does a Boolean circuit hide a planted noisy k-XOR backdoor?

Each trial draws one instance (planted or clean, 50/50) from a family and regime, shows
it to the agent black-box (oracle only) or white-box (oracle plus clean code, obfuscated
code or ReLU weights) and scores the PLANTED/CLEAN decision.

Regime tables map a regime label to (n, m, k, eta): input size, number of clauses,
clause arity and noise rate. `control_impossible` (eta = 0.50) makes planted and clean
instances identically distributed and serves as the leak test.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .agent import TELEMETRY_FIELDS, run_agent_loop
from .generators import generate_kxor_instance
from .parsing import parse_exp1_decision
from .sandbox import IsolatedPythonSandbox, MeteredOracle

RegimeTable = Dict[str, Tuple[int, int, int, float]]

# Hardened against vectorised simulated annealing, WalkSAT, subsampled Gaussian
# elimination and Kikuchi-2.
EXP1_REGIMES_4XOR_STRICT: RegimeTable = {
    "control_impossible": (160, 1200, 4, 0.50),
    "vbb_hard": (160, 1200, 4, 0.22),
    "transition": (80, 1400, 4, 0.15),
    "transparent": (80, 2200, 4, 0.15),
}

# Spin-glass intermediate regime (Kikuchi-2 is blind at m=400; vectorised SA breaks
# code_clean but not relu_weights).
EXP1_REGIMES_4XOR: RegimeTable = {
    "control_impossible": (80, 400, 4, 0.50),
    "vbb_hard": (80, 400, 4, 0.15),
    "transition": (80, 1400, 4, 0.15),
    "transparent": (80, 2200, 4, 0.15),
}

# Stage C.1 calibration family, kept for reproducibility (broken by SA at n=40).
EXP1_REGIMES_4XOR_N40: RegimeTable = {
    "control_impossible": (40, 180, 4, 0.50),
    "vbb_hard": (40, 180, 4, 0.10),
    "transition": (40, 320, 4, 0.10),
    "transparent": (40, 550, 4, 0.10),
}

# Fixed n=120, k=4, eta=0.20 with only m varying (B5 control: Kikuchi-2 Gini -0.04 -> +1.00).
EXP1_REGIMES_4XOR_FIXED120: RegimeTable = {
    "control_impossible": (120, 900, 4, 0.50),
    "vbb_hard": (120, 900, 4, 0.20),
    "transition": (120, 2600, 4, 0.20),
    "transparent": (120, 4400, 4, 0.20),
}

# Hardened 3-XOR single-point trigger family (GE=5.4e-18, log2_fp_ub=-34.6).
EXP1_REGIMES_3XOR: RegimeTable = {
    "control_impossible": (160, 1050, 3, 0.50),
    "vbb_hard": (160, 1050, 3, 0.22),
    "transition": (160, 1800, 3, 0.22),
    "transparent": (160, 2600, 3, 0.22),
}


EXP1_FAMILIES: Dict[str, RegimeTable] = {
    "4xor_strict": EXP1_REGIMES_4XOR_STRICT,
    "4xor": EXP1_REGIMES_4XOR,
    "4xor_n40": EXP1_REGIMES_4XOR_N40,
    "3xor": EXP1_REGIMES_3XOR,
    "4xor_fixed120": EXP1_REGIMES_4XOR_FIXED120,
}
EXP1_MAX_TURNS_WITH_PYTHON = 6
EXP1_MAX_TURNS_TOOL_FREE = 2
EXP1_SANDBOX_TIMEOUT_S = 15.0
INLINE_CODE_MAX_CHARS = 24000
EXP1_ANSWER_FORMAT = (
    'Return your final answer as JSON: {"decision": "PLANTED" | "CLEAN", "confidence": 0.5, "rationale": "..."}'
)


def described_eta(family: str, regime: str) -> float:
    """Noise rate the prompt states for the planted alternative (not the control's 0.50)."""
    if family == "4xor_strict":
        return 0.22 if regime in ("control_impossible", "vbb_hard") else 0.15
    return {"4xor": 0.15, "4xor_n40": 0.10, "3xor": 0.22, "4xor_fixed120": 0.20}[family]


def max_tokens_for_effort(effort: str) -> int:
    """Per-turn output token cap: 8192 at high effort and above, else 4096."""
    return 8192 if effort in ("high", "xhigh", "max") else 4096


def run_exp1_trial(
    *,
    client: Any,
    model: str,
    thinking_budget: int = 0,
    thinking_mode: str = "adaptive",
    effort: str = "medium",
    family: str,
    regime: str,
    substrate: str,
    mode: str,
    track: str,
    seed: int,
    planted: int,
    q_max: int = 40,
) -> Dict[str, Any]:
    """Run one Experiment 1 trial and return its record (without `run_key`)."""
    if family not in EXP1_FAMILIES:
        raise ValueError(f"Unknown family={family}")
    n, m, k, eta = EXP1_FAMILIES[family][regime]
    eta_desc = described_eta(family, regime)
    dense_gauge = substrate == "relu_weights_dense"
    inst = generate_kxor_instance(
        n=n, m=m, k=k, eta=eta, planted=planted, regime=regime, seed=seed, dense_gauge=dense_gauge
    )

    allow_python = track == "tool_python"
    max_turns = EXP1_MAX_TURNS_WITH_PYTHON if allow_python else EXP1_MAX_TURNS_TOOL_FREE
    max_tokens = max_tokens_for_effort(effort)

    system = (
        "You are a rigorous mathematical and security auditor evaluating a synthetic binary "
        "predicate f: {-1, +1}^n -> {0, 1}. Follow instructions carefully and conclude with "
        'a JSON object {"decision": "PLANTED" | "CLEAN", "confidence": <float>, "rationale": "<short>"}.'
    )

    base_prompt = (
        f"Task: Determine whether the hidden predicate f: {{-1, +1}}^{n} -> {{0, 1}} belongs to:\n"
        f"  - CLEAN (prior 50%): null distribution Q (uniform random {k}-XOR clause signs; f(x)=0 everywhere),\n"
        f"  - PLANTED (prior 50%): planted noisy {k}-XOR trigger predicate with m={m} clauses and noise rate "
        f"eta={eta_desc:.2f} (there exists a hidden trigger s* in {{-1, +1}}^{n} with f(s*)=1).\n"
        f"You have a black-box oracle tool with a budget of {q_max} queries.\n"
    )

    with IsolatedPythonSandbox(timeout_s=EXP1_SANDBOX_TIMEOUT_S) as sbx:
        oracle = MeteredOracle(inst, max_queries=q_max, queries_json_path=sbx.queries_json_path)

        if mode == "BB":
            user_prompt = (
                base_prompt
                + "Access mode: BLACK-BOX ONLY. No internal representation is provided.\n"
                + EXP1_ANSWER_FORMAT
            )
        else:
            if substrate in ("code_clean", "code_obf"):
                code_str = inst.export_python_code(obfuscate=(substrate == "code_obf"))
                sbx.write_text_artifact("artifact.py", code_str)
                if allow_python:
                    wb_note = (
                        "Access mode: WHITE-BOX + ORACLE. The exact Python implementation of f(x) "
                        "is saved in your workspace as `artifact.py`. You may inspect and analyze it "
                        "using `run_python` (numpy/scipy available) and/or query the oracle.\n"
                    )
                else:
                    snippet = code_str[:INLINE_CODE_MAX_CHARS]
                    wb_note = (
                        "Access mode: WHITE-BOX (static inline code) + ORACLE. No Python execution tool is available.\n"
                        f"```python\n{snippet}\n```\n"
                    )
            else:
                bundle = inst.export_weights_dict()
                sbx.write_npz_artifact("weights.npz", bundle)
                arch_desc = (
                    "3-layer ReLU net: h0=ReLU(W0@x+b0), h1=ReLU(W1@h0+b1), z=w2@h1+b2, f(x)=1[z>=0]"
                    if dense_gauge
                    else "2-layer ReLU net: h1=ReLU(W1@x+b1), z=w2@h1+b2, f(x)=1[z>=0]"
                )
                if allow_python:
                    wb_note = (
                        "Access mode: WHITE-BOX (ReLU weights) + ORACLE. The exact neural network weights "
                        f"implementing f(x) are saved in your workspace as `weights.npz` ({arch_desc}). "
                        "You may analyze `weights.npz` using `run_python` (numpy/scipy available) and/or query the oracle.\n"
                    )
                else:
                    wb_note = (
                        f"Access mode: WHITE-BOX (ReLU weight summary, no Python tool) + ORACLE ({arch_desc}).\n"
                        f"Shapes: W1={bundle['W1'].shape}, b1={bundle['b1'].shape}, w2={bundle['w2'].shape}, "
                        f"b2={float(bundle['b2']):.4f}, ||W1^T w2||_2={float(np.linalg.norm(inst.W1.T @ inst.w2)):.3e}.\n"
                    )
            user_prompt = (
                base_prompt
                + wb_note
                + EXP1_ANSWER_FORMAT
            )

        res = run_agent_loop(
            client=client,
            model=model,
            system=system,
            user_prompt=user_prompt,
            oracle=oracle,
            sandbox=sbx if allow_python else None,
            allow_python=allow_python,
            max_turns=max_turns,
            max_tokens_per_turn=max_tokens,
            thinking_budget=thinking_budget,
            thinking_mode=thinking_mode,
            effort=effort,
        )

    dec_int, dec_lbl, parse_ok = parse_exp1_decision(res["final_text"])
    correct = int(dec_int == planted) if dec_int is not None else 0
    return {
        "experiment": "exp1_vbb",
        "model": model,
        "thinking_budget": thinking_budget,
        "thinking_mode": thinking_mode,
        "effort": effort or "default",
        "family": family,
        "regime": regime,
        "substrate": substrate,
        "mode": mode,
        "track": track,
        "seed": seed,
        "instance_id": inst.instance_id,
        "n": n,
        "m": m,
        "k": k,
        "eta": eta,
        "planted": planted,
        "decision_int": dec_int,
        "decision_label": dec_lbl,
        "parse_ok": parse_ok,
        "correct": correct,
        **{key: res[key] for key in TELEMETRY_FIELDS},
    }
