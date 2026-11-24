"""Command line driver: build the trial schedule, run it and append records to a JSONL log.

    python -m blackwell_inversion --dry-run --experiment exp2 --exp2-seeds 1 --out dry.jsonl

`--dry-run` uses the zero-cost mock client. Otherwise `--provider` picks the client; with
the default `auto` it is chosen from the model ids: Gemini/Gemma ids use the Gemini client,
anything else the Anthropic client. `--provider openai --base-url URL` talks to any
OpenAI-compatible server, and `--client module:factory` plugs in a client of your own.
`--preset NAME` sets every option of one of the paper's conditions (see `presets`).
A rerun with the same --out file skips every trial already logged.

A trial that raises an error is reported on one line (`--debug` adds the traceback), is
not written to the log and is retried by the next run; the run goes on with the next
trial, and stops after `--max-consecutive-errors` errors in a row (a broken setup).
Ctrl-C stops scheduling: queued trials are cancelled, requests in flight finish or time
out, and every trial already written stays in the log.

Exit status: 0 when every scheduled trial is in the log (or was already), 1 when a trial
failed or the run was stopped by the consecutive-error guard, 2 for a usage error, 3 when
the spend cap (or an HTTP 402) stopped the run without trial errors, 130 after Ctrl-C.
Statuses 1, 3 and 130 leave a resumable log.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import importlib.util
import os
import shlex
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .clients import AnthropicClient, BudgetExceededError, GeminiClient, MockDryRunClient, ModelClient, OpenAICompatClient
from .clients.openai_compat import MAX_TOKENS_FIELDS
from .exp1 import EXP1_FAMILIES, run_exp1_trial
from .exp2 import EXP2_PROTOCOLS, run_exp2_trial
from .presets import PRESET_DESCRIPTIONS, PRESETS, PresetError, apply_preset, explicit_options
from .schedule import Job, append_jsonl, build_exp1_schedule, build_exp2_schedule, load_completed_keys

GOOGLE_MODEL_PREFIXES = ("gemini-", "gemma-")
MOCK_MODEL_PREFIX = "mock-"
PROGRESS_EVERY = 25
PROVIDERS = ("auto", "anthropic", "gemini", "openai", "mock")
DEFAULT_MAX_CONSECUTIVE_ERRORS = 5
# Exit statuses of `main` (2 is argparse's usage error).
EXIT_OK = 0
EXIT_TRIAL_ERRORS = 1
EXIT_SPEND_CAP = 3
EXIT_INTERRUPTED = 130


class ClientLoadError(ValueError):
    """A --client spec cannot be loaded or its factory does not return a ModelClient."""


class _RunStopped(Exception):
    """Raised instead of a model request once a threaded run has been stopped."""


class _StopGate:
    """Passes requests to the client until the run is stopped, then refuses them.

    Used in threaded runs, so that a trial in flight ends at its next request after the
    spend cap, the error guard or Ctrl-C has stopped the run.
    """

    def __init__(self, client: ModelClient, stop_event: threading.Event) -> None:
        self._client = client
        self._stop_event = stop_event

    def create_message(self, **kwargs: Any) -> Dict[str, Any]:
        if self._stop_event.is_set():
            raise _RunStopped()
        return self._client.create_message(**kwargs)


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
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default=None,
        help="Set every condition option of one of the paper's conditions (see --list-presets).",
    )
    parser.add_argument("--list-presets", action="store_true", help="List the presets and the options they set, and exit.")
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default="auto",
        help="Model client: auto (mock with --dry-run, Gemini for Gemini/Gemma ids, else Anthropic), anthropic, "
        "gemini, openai (any OpenAI-compatible server, needs --base-url) or mock (same as --dry-run).",
    )
    parser.add_argument(
        "--client",
        default="",
        metavar="MODULE:FACTORY",
        help="Custom client: 'package.module:factory' or 'path/to/file.py:factory'; factory(args) must return a ModelClient.",
    )
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL, e.g. http://127.0.0.1:11434/v1 (--provider openai).")
    parser.add_argument(
        "--api-key-env",
        default="",
        metavar="VAR",
        help="Environment variable holding the API key for --provider openai; empty (default) sends no key.",
    )
    parser.add_argument("--input-price-per-mtok", type=float, default=0.0, help="USD per million input tokens (--provider openai).")
    parser.add_argument("--output-price-per-mtok", type=float, default=0.0, help="USD per million output tokens (--provider openai).")
    parser.add_argument(
        "--max-tokens-field",
        choices=MAX_TOKENS_FIELDS,
        default="max_tokens",
        help="Request field for the per-turn output token cap (--provider openai).",
    )
    parser.add_argument("--request-timeout-s", type=float, default=600.0, help="Seconds to wait for one reply (--provider openai).")
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
        help="Effort level (Anthropic output_config.effort; Gemini 3/4 thinking level; OpenAI-compatible "
        "reasoning_effort, which --effort '' leaves out).",
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
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_ERRORS,
        metavar="N",
        help=f"Stop the run after N trial errors in a row (default {DEFAULT_MAX_CONSECUTIVE_ERRORS}; 0 never stops).",
    )
    parser.add_argument("--debug", action="store_true", help="Print the traceback of every trial error.")
    parser.add_argument("--out", default="results.jsonl", help="Output JSONL log (resumable).")
    return parser


def load_custom_client(spec: str, args: argparse.Namespace) -> ModelClient:
    """Import `module:factory` (or `file.py:factory`), call factory(args) and check the result."""
    module_part, sep, attr = spec.rpartition(":")
    if not sep or not module_part or not attr:
        raise ClientLoadError(f"--client {spec!r} must have the form package.module:factory or path/to/file.py:factory.")
    if module_part.endswith(".py"):
        path = Path(module_part)
        if not path.is_file():
            raise ClientLoadError(f"--client: file {module_part!r} not found.")
        file_spec = importlib.util.spec_from_file_location(f"_custom_client_{path.stem}", path)
        if file_spec is None or file_spec.loader is None:
            raise ClientLoadError(f"--client: cannot load {module_part!r} as a module.")
    else:
        # A module path is resolved from the working directory, as with `python -m`, also
        # when the installed `blackwell-inversion` command (whose sys.path lacks it) runs.
        cwd = os.getcwd()
        if not any(entry in ("", ".") or os.path.abspath(entry) == cwd for entry in sys.path):
            sys.path.insert(0, cwd)
    try:
        if module_part.endswith(".py"):
            module = importlib.util.module_from_spec(file_spec)
            file_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_part)
    except ImportError as exc:
        raise ClientLoadError(f"--client: cannot import {module_part!r}: {exc}") from exc
    except Exception as exc:  # the module itself fails: SyntaxError, NameError, ...
        raise ClientLoadError(f"--client: loading {module_part!r} failed: {type(exc).__name__}: {exc}") from exc
    factory = getattr(module, attr, None)
    if not callable(factory):
        raise ClientLoadError(f"--client: {module_part!r} has no callable {attr!r}.")
    try:
        client = factory(args)
    except Exception as exc:
        raise ClientLoadError(f"--client: {spec}(args) raised {type(exc).__name__}: {exc}") from exc
    if not isinstance(client, ModelClient):
        raise ClientLoadError(
            f"--client: {spec} returned {type(client).__name__}, not a blackwell_inversion.clients.ModelClient."
        )
    return client


def make_client(args: argparse.Namespace, models: Sequence[str]) -> ModelClient:
    """The client for this run: --client, else --provider; `auto` keeps the original choice.

    `auto` means the mock client for --dry-run, the Gemini client if every model is a
    Google model, else the Anthropic client.
    """
    if args.client:
        return load_custom_client(args.client, args)
    if args.dry_run or args.provider == "mock":
        return MockDryRunClient()
    if args.provider == "openai":
        return OpenAICompatClient(
            args.base_url,
            api_key_env=args.api_key_env,
            max_cost_usd=args.max_cost_usd,
            input_price_per_mtok=args.input_price_per_mtok,
            output_price_per_mtok=args.output_price_per_mtok,
            max_tokens_field=args.max_tokens_field,
            timeout_s=args.request_timeout_s,
        )
    if args.provider == "gemini" or (
        args.provider == "auto" and all(m.lower().startswith(GOOGLE_MODEL_PREFIXES) for m in models)
    ):
        return GeminiClient(max_cost_usd=args.max_cost_usd, min_interval_s=args.min_interval_s)
    return AnthropicClient(max_cost_usd=args.max_cost_usd)


def check_client_options(args: argparse.Namespace, given: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """A message if the client options contradict each other, else None.

    `given` holds the options given on the command line (`presets.explicit_options`).
    """
    if args.client and (args.provider != "auto" or args.dry_run):
        return "--client cannot be combined with --provider or --dry-run."
    if args.dry_run and args.provider not in ("auto", "mock"):
        return f"--dry-run uses the mock client; it cannot be combined with --provider {args.provider}."
    if args.provider == "openai" and not args.base_url:
        return "--provider openai needs --base-url (for example http://127.0.0.1:11434/v1)."
    if args.provider == "openai" and given is not None and "models" not in given:
        return "--provider openai needs --models: the model name(s) your server expects."
    if args.base_url and args.provider != "openai":
        return "--base-url is used only with --provider openai."
    return None


def _list_presets() -> int:
    for name, opts in PRESETS.items():
        print(f"{name}: {PRESET_DESCRIPTIONS[name]}")
        flags = []
        for dest, value in opts.items():
            flag = "--" + dest.replace("_", "-")
            if isinstance(value, bool):
                flags.append(flag if value else f"(no {flag})")
            else:
                flags.append(f"{flag} {shlex.quote(str(value))}")
        print("    " + " ".join(flags))
    return 0


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
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_models:
        return _list_models()
    if args.list_presets:
        return _list_presets()
    given = explicit_options(build_parser, argv)
    if args.preset:
        try:
            for note in apply_preset(args, given):
                print(f"[PRESET] {note}", file=sys.stderr)
        except PresetError as exc:
            parser.error(str(exc))
    problem = check_client_options(args, given)
    if problem:
        parser.error(problem)
    if args.max_consecutive_errors < 0:
        parser.error("--max-consecutive-errors must be 0 (never stop) or positive.")
    if args.provider == "mock":
        args.dry_run = True

    models = _csv(args.models)
    if args.dry_run:
        models = [m if m.startswith(MOCK_MODEL_PREFIX) else f"{MOCK_MODEL_PREFIX}{m}" for m in models]
    try:
        client = make_client(args, models)
    except (ClientLoadError, ValueError) as exc:
        parser.error(str(exc))

    out_path = Path(args.out)
    done = load_completed_keys(out_path)
    for extra_path in _csv(args.resume_from):
        done |= load_completed_keys(Path(extra_path))

    all_jobs = build_jobs(args, models)
    pending = [(kind, job) for kind, job in all_jobs if job["run_key"] not in done]
    skipped = len(all_jobs) - len(pending)

    run = _Run(client, args, out_path, len(pending))
    if max(1, int(args.workers)) == 1:
        _run_serial(run, pending)
    else:
        _run_parallel(run, pending)

    print(
        f"Done. executed={run.executed}, skipped_resumed={skipped}, errors={run.errors}, "
        f"total_cost_usd=${client.total_cost_usd:.4f}, out={out_path}"
    )
    status = run.exit_status()
    if status != EXIT_OK:
        print(f"Exit status {status}: the log is incomplete; rerun the same command to resume.", file=sys.stderr)
    return status


class _Run:
    """Outcome accounting of one invocation, shared by the serial and the threaded mode.

    `finish_*` may be called from worker threads; the lock serialises the counters, the
    log writes and the messages.
    """

    def __init__(self, client: ModelClient, args: argparse.Namespace, out_path: Path, n_pending: int) -> None:
        self.client = client
        self.args = args
        self.out_path = out_path
        self.n_pending = n_pending
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self.executed = 0
        self.errors = 0
        self.consecutive_errors = 0
        self.spend_cap = False
        self.aborted = False
        self.interrupted = False

    def finish_record(self, rec: Dict[str, Any]) -> None:
        with self._lock:
            append_jsonl(self.out_path, rec)
            self.executed += 1
            self.consecutive_errors = 0
            if int(self.args.workers) > 1 and (self.executed % PROGRESS_EVERY == 0 or self.executed == self.n_pending):
                print(
                    f"  progress: {self.executed}/{self.n_pending} (errors={self.errors}, cost=${self.client.total_cost_usd:.4f})",
                    flush=True,
                )

    def finish_error(self, run_key: str, exc: BaseException) -> None:
        with self._lock:
            self.errors += 1
            self.consecutive_errors += 1
            message = " ".join(str(exc).split()) or type(exc).__name__
            print(f"[TRIAL ERROR] {run_key}: {message}", file=sys.stderr)
            if self.args.debug:
                traceback.print_exception(exc, file=sys.stderr)
            limit = int(self.args.max_consecutive_errors)
            if limit and self.consecutive_errors >= limit and not self.stop_event.is_set():
                self.aborted = True
                self.stop_event.set()
                print(
                    f"[STOPPED — {limit} CONSECUTIVE TRIAL ERRORS] The setup looks broken (see the errors above); "
                    "fix it and rerun the same command to resume (--max-consecutive-errors changes the limit).",
                    file=sys.stderr,
                )

    def finish_spend_cap(self, exc: BudgetExceededError) -> None:
        with self._lock:
            if not self.stop_event.is_set():
                print(f"[STOPPED — BUDGET CAP] {exc}", file=sys.stderr)
            self.spend_cap = True
            self.stop_event.set()

    def exit_status(self) -> int:
        if self.interrupted:
            return EXIT_INTERRUPTED
        if self.errors or self.aborted:
            return EXIT_TRIAL_ERRORS
        if self.spend_cap:
            return EXIT_SPEND_CAP
        return EXIT_OK


def _run_serial(run: _Run, pending: List[Tuple[str, Job]]) -> None:
    for kind, job in pending:
        if run.stop_event.is_set():
            break
        try:
            rec = run_trial(kind, job, run.client, run.args)
        except BudgetExceededError as exc:
            run.finish_spend_cap(exc)
            break
        except KeyboardInterrupt:
            run.interrupted = True
            print("[INTERRUPTED] Stopped; the trial in progress is not written. Rerun the same command to resume.", file=sys.stderr)
            break
        except Exception as exc:
            run.finish_error(job["run_key"], exc)
            continue
        run.finish_record(rec)


def _run_parallel(run: _Run, pending: List[Tuple[str, Job]]) -> None:
    """Run trials on a thread pool with the same outcomes as `_run_serial`.

    Once the run is stopped (spend cap, error guard, Ctrl-C), no trial starts and a trial
    in flight ends at its next model request without a record.
    """
    gate = _StopGate(run.client, run.stop_event)

    def run_one(kind: str, job: Job) -> None:
        if run.stop_event.is_set():
            return
        try:
            rec = run_trial(kind, job, gate, run.args)  # type: ignore[arg-type]
        except _RunStopped:
            return
        except BudgetExceededError as exc:
            run.finish_spend_cap(exc)
            return
        except Exception as exc:
            run.finish_error(job["run_key"], exc)
            return
        run.finish_record(rec)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=int(run.args.workers))
    futures: List[concurrent.futures.Future[None]] = []
    try:
        for kind, job in pending:
            futures.append(pool.submit(run_one, kind, job))
        concurrent.futures.wait(futures)
    except KeyboardInterrupt:
        run.interrupted = True
        run.stop_event.set()
        cancelled = sum(1 for fut in futures if fut.cancel())
        in_flight = sum(1 for fut in futures if not fut.done())
        print(
            f"[INTERRUPTED] {cancelled} queued trials cancelled; waiting for {in_flight} trials in flight to end at their "
            "current request (or its timeout). Trials already written stay in the log; rerun the same command to resume.",
            file=sys.stderr,
            flush=True,
        )
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
