"""Command line driver: build the trial schedule, run it and append records to a JSONL log.

    python -m blackwell_inversion --dry-run --experiment exp2 --exp2-seeds 1 --out dry.jsonl

`--dry-run` uses the zero-cost mock client. Otherwise the provider is chosen from the
model ids: Gemini/Gemma ids use the Gemini client, anything else the Anthropic client.
A rerun with the same --out file skips every trial already logged.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .clients import AnthropicClient, BudgetExceededError, GeminiClient, MockDryRunClient, ModelClient
from .exp1 import EXP1_FAMILIES, run_exp1_trial
from .exp2 import EXP2_PROTOCOLS, run_exp2_trial
from .schedule import Job, append_jsonl, build_exp1_schedule, build_exp2_schedule, load_completed_keys

GOOGLE_MODEL_PREFIXES = ("gemini-", "gemma-")
MOCK_MODEL_PREFIX = "mock-"
PROGRESS_EVERY = 25


def _csv(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blackwell-inversion",
        description="Run the auditor-agent experiments (Exp 1: C_kXOR backdoor detection, "
        "Exp 2: C_inv rule recovery) and append one JSON record per trial.",
    )
    parser.add_argument("--list-models", action="store_true", help="List the Anthropic models visible to the API key and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Run the full tool/sandbox pipeline with the zero-cost mock client.")
    parser.add_argument("--experiment", choices=("exp1", "exp2", "all"), default="all", help="Which experiment to run.")
    parser.add_argument("--models", default="claude-haiku-5-5,claude-sonnet-5-5", help="Comma-separated model ids.")
    parser.add_argument(
        "--thinking-mode",
        choices=("adaptive", "default", "disabled", "budget"),
        default="adaptive",
        help="Reasoning mode (Anthropic thinking; Gemini maps 'disabled' to the minimal thinking level).",
    )
    parser.add_argument(
        "--effort",
        choices=("", "low", "medium", "high", "xhigh", "max"),
        default="medium",
        help="Effort level (Anthropic output_config.effort; Gemini 3/4 thinking level).",
    )
    parser.add_argument("--thinking-budget", type=int, default=0, help="Thinking token budget (used with --thinking-mode budget, and by Gemini 2.5).")
    parser.add_argument("--family", choices=tuple(EXP1_FAMILIES), default="4xor_strict", help="Exp 1 instance family.")
    parser.add_argument("--task-type", choices=("majority3", "dtree2", "dlist3"), default="majority3", help="Exp 2 task type.")
    parser.add_argument("--regimes", default="control_impossible,vbb_hard,transition,transparent", help="Comma-separated Exp 1 regimes.")
    parser.add_argument("--substrates", default="code_clean,code_obf,relu_weights", help="Comma-separated Exp 1 WB substrates.")
    parser.add_argument("--tracks", default="tool_python", help="Comma-separated Exp 1 tracks (tool_free, tool_python).")
    parser.add_argument("--exp1-pairs", type=int, default=20, help="(planted, clean) pairs per Exp 1 cell (2 * pairs trials).")
    parser.add_argument("--exp1-seed-offset", type=int, default=1000, help="Seed offset for Exp 1 instances.")
    parser.add_argument("--exp1-skip-bb", action="store_true", help="Skip the BB cells in the Exp 1 schedule.")
    parser.add_argument("--variants", default="wb_clean,wb_inv_code,wb_inv_relu", help="Comma-separated Exp 2 WB variants.")
    parser.add_argument(
        "--budget-tracks",
        default="matched_total_budget,matched_query_free_code",
        help="Comma-separated Exp 2 budget tracks.",
    )
    parser.add_argument("--exp2-seeds", type=int, default=30, help="Matched seeds per Exp 2 cell.")
    parser.add_argument("--skip-bb", action="store_true", help="Skip the BB trials in the Exp 2 schedule.")
    parser.add_argument("--only-bb", action="store_true", help="Run only the BB trials of the Exp 2 schedule.")
    parser.add_argument("--max-tokens-override", type=int, default=0, help="Per-turn output token cap for Exp 2 (e.g. 8192).")
    parser.add_argument(
        "--protocol",
        default="standard",
        choices=list(EXP2_PROTOCOLS),
        help="Exp 2 protocol: standard, verify_first (prompt-level), harness_check (answer tested by the "
        "harness) or harness_check_tools (same check, revision turns with tools).",
    )
    parser.add_argument(
        "--log-tool-inputs",
        action="store_true",
        help="Store the Python code of each tool call and the queried inputs in every Exp 2 record.",
    )
    parser.add_argument("--resume-from", default="", help="Comma-separated extra JSONL files whose run keys are skipped (read only).")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent worker threads.")
    parser.add_argument("--min-interval-s", type=float, default=0.0, help="Minimum seconds between Gemini API calls (for tight rate limits).")
    parser.add_argument("--max-cost-usd", type=float, default=35.0, help="Hard spend cap in USD for this run.")
    parser.add_argument("--out", default="pilot_results.jsonl", help="Output JSONL log (resumable).")
    return parser


def make_client(args: argparse.Namespace, models: Sequence[str]) -> ModelClient:
    """Mock client for --dry-run, Gemini client if every model is a Google model, else Anthropic."""
    if args.dry_run:
        return MockDryRunClient()
    if all(m.lower().startswith(GOOGLE_MODEL_PREFIXES) for m in models):
        return GeminiClient(max_cost_usd=args.max_cost_usd, min_interval_s=args.min_interval_s)
    return AnthropicClient(max_cost_usd=args.max_cost_usd)


def build_jobs(args: argparse.Namespace, models: Sequence[str]) -> List[Tuple[str, Job]]:
    """The full schedule as (experiment, job) pairs, Exp 1 first."""
    jobs: List[Tuple[str, Job]] = []
    if args.experiment in ("exp1", "all"):
        jobs += [
            ("exp1", job)
            for job in build_exp1_schedule(
                models,
                args.family,
                _csv(args.regimes),
                _csv(args.substrates),
                _csv(args.tracks),
                args.exp1_pairs,
                seed_offset=args.exp1_seed_offset,
                thinking_mode=args.thinking_mode,
                effort=args.effort,
                skip_bb=args.exp1_skip_bb,
            )
        ]
    if args.experiment in ("exp2", "all"):
        jobs += [
            ("exp2", job)
            for job in build_exp2_schedule(
                models,
                _csv(args.variants),
                _csv(args.budget_tracks),
                args.exp2_seeds,
                thinking_mode=args.thinking_mode,
                effort=args.effort,
                task_type=args.task_type,
                skip_bb=args.skip_bb,
                only_bb=args.only_bb,
                max_tokens_override=args.max_tokens_override,
                protocol=args.protocol,
            )
        ]
    return jobs


def run_trial(kind: str, job: Job, client: ModelClient, args: argparse.Namespace) -> Dict[str, Any]:
    """Run one scheduled trial and return its complete record."""
    if kind == "exp1":
        rec = run_exp1_trial(
            client=client,
            model=job["model"],
            thinking_budget=args.thinking_budget,
            thinking_mode=args.thinking_mode,
            effort=args.effort,
            family=job["family"],
            regime=job["regime"],
            substrate=job["substrate"],
            mode=job["mode"],
            track=job["track"],
            seed=job["seed"],
            planted=job["planted"],
        )
    else:
        rec = run_exp2_trial(
            client=client,
            model=job["model"],
            thinking_budget=args.thinking_budget,
            thinking_mode=args.thinking_mode,
            effort=args.effort,
            variant=job["variant"],
            mode=job["mode"],
            budget_track=job["budget_track"],
            seed=job["seed"],
            task_type=job.get("task_type", "majority3"),
            max_tokens_override=args.max_tokens_override,
            protocol=job.get("protocol", "standard"),
            log_tool_inputs=args.log_tool_inputs,
        )
    if args.max_tokens_override > 0:
        rec["max_tokens_override"] = int(args.max_tokens_override)
    rec["run_key"] = job["run_key"]
    return rec


def _list_models() -> int:
    client = AnthropicClient(max_cost_usd=0.0)
    models = client.list_models()
    print(f"Found {len(models)} Anthropic models:")
    for m in models:
        print(f"  - {m.get('id')}  (display={m.get('display_name', '')}, created={m.get('created_at', '')})")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_models:
        return _list_models()

    models = _csv(args.models)
    if args.dry_run:
        models = [m if m.startswith(MOCK_MODEL_PREFIX) else f"{MOCK_MODEL_PREFIX}{m}" for m in models]
    client = make_client(args, models)

    out_path = Path(args.out)
    done = load_completed_keys(out_path)
    for extra_path in _csv(args.resume_from):
        done |= load_completed_keys(Path(extra_path))

    all_jobs = build_jobs(args, models)
    pending = [(kind, job) for kind, job in all_jobs if job["run_key"] not in done]
    skipped = len(all_jobs) - len(pending)
    executed = 0
    errors = 0

    if max(1, int(args.workers)) == 1:
        # Serial mode stops at the first trial error as well as at the spend cap.
        try:
            for kind, job in pending:
                append_jsonl(out_path, run_trial(kind, job, client, args))
                executed += 1
        except BudgetExceededError as exc:
            print(f"[STOPPED — BUDGET CAP] {exc}", file=sys.stderr)
    else:
        executed, errors = _run_parallel(pending, client, args, out_path)

    print(
        f"Done. executed={executed}, skipped_resumed={skipped}, errors={errors}, "
        f"total_cost_usd=${client.total_cost_usd:.4f}, out={out_path}"
    )
    return 0


def _run_parallel(
    pending: List[Tuple[str, Job]], client: ModelClient, args: argparse.Namespace, out_path: Path
) -> Tuple[int, int]:
    """Run trials on a thread pool; a trial error is counted and skipped, the spend cap stops new trials."""
    stop_event = threading.Event()
    write_lock = threading.Lock()
    executed = 0
    errors = 0

    def run_one(kind: str, job: Job) -> Optional[Dict[str, Any]]:
        if stop_event.is_set():
            return None
        return run_trial(kind, job, client, args)

    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = {pool.submit(run_one, kind, job): job["run_key"] for kind, job in pending}
        for fut in concurrent.futures.as_completed(futures):
            run_key = futures[fut]
            try:
                rec = fut.result()
            except BudgetExceededError as exc:
                if not stop_event.is_set():
                    stop_event.set()
                    print(f"[STOPPED — BUDGET CAP] {exc}", file=sys.stderr)
                continue
            except Exception as exc:
                errors += 1
                print(f"[TRIAL ERROR] {run_key}: {exc}", file=sys.stderr)
                continue
            if rec is None:
                continue
            with write_lock:
                append_jsonl(out_path, rec)
                executed += 1
                if executed % PROGRESS_EVERY == 0 or executed == len(pending):
                    print(
                        f"  progress: {executed}/{len(pending)} (errors={errors}, cost=${client.total_cost_usd:.4f})",
                        flush=True,
                    )
    return executed, errors
