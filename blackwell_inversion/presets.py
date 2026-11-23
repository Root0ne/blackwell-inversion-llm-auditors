"""Named command line presets for the paper's main conditions.

A preset fixes every option that defines a condition (experiment, task, protocol,
budget track, variants, seeds, reasoning settings), so a run with your own model has
run keys of exactly the form of the paper's records:

    python -m blackwell_inversion --preset paper-exp2-main --provider openai \
        --base-url http://127.0.0.1:11434/v1 --models my-model --out my.jsonl

The values were read off the locked logs in `data/main/` and are checked against them
by `tests/test_presets.py`:

    paper-exp2-main  Exp 2, Majority-3, standard protocol, medium effort, 5/5 turns
                     (`matched_medium_budget`), seeds 5000-5024, black box plus the five
                     white-box variants: 150 trials per model. The paper's main Exp 2
                     condition; 13 of its 15 models ran it with these 25 seeds (the
                     Gemma models with 15 and 10, Gemini 3.1 Flash-Lite with 20).
    paper-exp1       Exp 1, hardened 4-XOR family, Python tool, hard and solvable
                     regimes, black box plus clean code, sparse weights and dense
                     weights, 12 planted/clean pairs: 192 trials per model. The design
                     Opus 5.5 and Fable 5.2 ran; some paper cells have more pairs.
    smoke            Seed 5000 of paper-exp2-main, black box, clean code and hard-guard
                     code: 3 trials. Its run keys are a subset of paper-exp2-main, so a
                     later paper-exp2-main run with the same --out resumes after them.

Options a preset fixes may not be given on the command line with another value, except
the reasoning settings `--effort` and `--thinking-mode`: a model or server without
reasoning controls can run a preset with `--effort ""`. Its run keys then read
`ef=default`, and the summary marks such cells as differing from the paper's setting.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, List, Sequence

EXP2_MAIN_VARIANTS = "wb_clean,wb_decoy_trivial_guard,wb_decoy_long_refutable_guard,wb_inv_code,wb_inv_relu"

_EXP2_MAIN: Dict[str, Any] = {
    "experiment": "exp2",
    "task_type": "majority3",
    "protocol": "standard",
    "thinking_mode": "adaptive",
    "effort": "medium",
    "budget_tracks": "matched_medium_budget",
    "variants": EXP2_MAIN_VARIANTS,
    "exp2_seeds": 25,
    "skip_bb": False,
    "only_bb": False,
    "max_tokens_override": 0,
}

PRESETS: Dict[str, Dict[str, Any]] = {
    "paper-exp2-main": dict(_EXP2_MAIN),
    "paper-exp1": {
        "experiment": "exp1",
        "family": "4xor_strict",
        "tracks": "tool_python",
        "regimes": "vbb_hard,transparent",
        "substrates": "code_clean,relu_weights,relu_weights_dense",
        "exp1_pairs": 12,
        "exp1_seed_offset": 1000,
        "exp1_skip_bb": False,
        "thinking_mode": "adaptive",
        "effort": "medium",
    },
    "smoke": {**_EXP2_MAIN, "variants": "wb_clean,wb_inv_code", "exp2_seeds": 1},
}

PRESET_DESCRIPTIONS: Dict[str, str] = {
    "paper-exp2-main": "Exp 2 main condition (Majority-3, standard, medium effort, 5/5 turns, seeds 5000-5024, BB + 5 WB variants); 150 trials per model",
    "paper-exp1": "Exp 1 core design (4xor_strict, Python tool, hard + solvable regimes, BB + 3 WB substrates, 12 pairs); 192 trials per model",
    "smoke": "3 trials of paper-exp2-main (seed 5000: BB, clean code, hard-guard code); checks that a model and server work",
}

# Options a preset fixes that may still be overridden explicitly.
OVERRIDABLE = ("effort", "thinking_mode")


class PresetError(ValueError):
    """A preset name is unknown or a command line option contradicts the preset."""


def explicit_options(build_parser: Callable[[], argparse.ArgumentParser], argv: Sequence[str]) -> Dict[str, Any]:
    """The options given on the command line, as {dest: value}; defaults are left out.

    `build_parser` makes a fresh parser; every default of it is replaced by a marker
    object, so the options still holding the marker after parsing were not given.
    """
    unset = object()
    probe = build_parser()
    probe.set_defaults(**{dest: unset for dest in vars(probe.parse_args([]))})
    return {dest: value for dest, value in vars(probe.parse_args(list(argv))).items() if value is not unset}


def apply_preset(args: argparse.Namespace, explicit: Dict[str, Any]) -> List[str]:
    """Set the preset's options on `args`; return notes about overridden reasoning settings."""
    if args.preset not in PRESETS:
        raise PresetError(f"Unknown preset {args.preset!r}; choose from {', '.join(PRESETS)}.")
    notes: List[str] = []
    for dest, value in PRESETS[args.preset].items():
        if dest in explicit and explicit[dest] != value:
            if dest in OVERRIDABLE:
                notes.append(
                    f"--{dest.replace('_', '-')} {explicit[dest]!r} overrides the preset's {value!r}; "
                    "run keys will differ from the paper's records."
                )
                continue
            raise PresetError(
                f"--preset {args.preset} fixes --{dest.replace('_', '-')}={value!r}; "
                f"it cannot be combined with {explicit[dest]!r}."
            )
        setattr(args, dest, value)
    return notes
