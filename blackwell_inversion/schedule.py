"""Trial schedules, run keys and the resumable JSONL log.

A schedule is a list of jobs, one per trial, each with a unique `run_key` that encodes
the experiment, model, reasoning settings and every condition label. A run appends one
JSON line per finished trial; on restart, jobs whose key is already in the output file
(or in any --resume-from file) are skipped.

Run key formats (the tags in brackets appear only when they differ from the default):

    exp1|<model>|tm=<thinking_mode>|ef=<effort>|<family>|<track>|<regime>|<mode>|<substrate>|s<seed>|b<planted>
    exp2|<model>|tm=<thinking_mode>|ef=<effort>[|tt=<task_type>][|mt=<max_tokens>][|pr=<protocol>]|<budget_track>|<mode>|<variant>|s<seed>

In Exp 2 the BB job carries `variant=wb_inv_code` and `blackbox` in its key; its instance
has the same hidden rule as every WB variant of the seed, so it pairs with all of them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

Job = Dict[str, Any]


def load_completed_keys(path: Path) -> set[str]:
    """Run keys already present in a JSONL log; unreadable lines are ignored."""
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "run_key" in rec:
                    done.add(str(rec["run_key"]))
            except json.JSONDecodeError:
                continue
    return done


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append one record as a JSON line, creating the parent directory if needed.

    If the file does not end in a newline (an earlier write was interrupted), the record
    starts on a new line, so it is not merged into the unreadable partial line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as fh:
        prefix = b""
        if fh.seek(0, 2) > 0:
            fh.seek(-1, 2)
            if fh.read(1) != b"\n":
                prefix = b"\n"
        fh.write(prefix + (json.dumps(record) + "\n").encode("utf-8"))


def build_exp1_schedule(
    models: Sequence[str],
    family: str,
    regimes: Sequence[str],
    substrates: Sequence[str],
    tracks: Sequence[str],
    n_pairs: int,
    seed_offset: int = 1000,
    thinking_mode: str = "adaptive",
    effort: str = "medium",
    skip_bb: bool = False,
) -> List[Job]:
    """Balanced Exp 1 schedule: n_pairs planted and n_pairs clean trials per cell.

    One BB cell per (model, track, regime) is shared by all WB substrates. Seeds are
    seed_offset + 100 * i + planted, so planted and clean trials never share a seed.
    """
    jobs: List[Job] = []
    ef_tag = f"|tm={thinking_mode}|ef={effort or 'default'}"
    for model in models:
        for track in tracks:
            for regime in regimes:
                mode_subs = ([] if skip_bb else [("BB", "blackbox")]) + [("WB", s) for s in substrates]
                for mode, sub in mode_subs:
                    for i in range(n_pairs):
                        for planted in (0, 1):
                            seed = seed_offset + 100 * i + planted
                            rkey = f"exp1|{model}{ef_tag}|{family}|{track}|{regime}|{mode}|{sub}|s{seed}|b{planted}"
                            jobs.append(
                                {
                                    "run_key": rkey,
                                    "model": model,
                                    "family": family,
                                    "track": track,
                                    "regime": regime,
                                    "mode": mode,
                                    "substrate": sub,
                                    "seed": seed,
                                    "planted": planted,
                                }
                            )
    return jobs


def build_exp2_schedule(
    models: Sequence[str],
    variants: Sequence[str],
    budget_tracks: Sequence[str],
    n_seeds: int,
    seed_offset: int = 5000,
    thinking_mode: str = "adaptive",
    effort: str = "medium",
    task_type: str = "majority3",
    skip_bb: bool = False,
    only_bb: bool = False,
    max_tokens_override: int = 0,
    protocol: str = "standard",
) -> List[Job]:
    """Matched-pair Exp 2 schedule: the same seeds for BB and every WB variant."""
    jobs: List[Job] = []
    ef_tag = f"|tm={thinking_mode}|ef={effort or 'default'}"
    tt_tag = "" if task_type == "majority3" else f"|tt={task_type}"
    mt_tag = "" if max_tokens_override <= 0 else f"|mt={max_tokens_override}"
    if protocol != "standard":
        mt_tag += f"|pr={protocol}"
    for model in models:
        for track in budget_tracks:
            for i in range(n_seeds):
                seed = seed_offset + i
                if not skip_bb:
                    # Any variant would do: all variants of a seed share the hidden rule.
                    rkey_bb = f"exp2|{model}{ef_tag}{tt_tag}{mt_tag}|{track}|BB|blackbox|s{seed}"
                    jobs.append(
                        {
                            "run_key": rkey_bb,
                            "model": model,
                            "budget_track": track,
                            "mode": "BB",
                            "variant": "wb_inv_code",
                            "seed": seed,
                            "task_type": task_type,
                            "protocol": protocol,
                        }
                    )
                if not only_bb:
                    for var in variants:
                        rkey_wb = f"exp2|{model}{ef_tag}{tt_tag}{mt_tag}|{track}|WB|{var}|s{seed}"
                        jobs.append(
                            {
                                "run_key": rkey_wb,
                                "model": model,
                                "budget_track": track,
                                "mode": "WB",
                                "variant": var,
                                "seed": seed,
                                "task_type": task_type,
                                "protocol": protocol,
                            }
                        )
    return jobs
