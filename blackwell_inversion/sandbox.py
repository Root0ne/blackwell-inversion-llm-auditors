"""Workspace in which agent-written Python runs, and the metered black-box oracle.

Anti-cheating and hygiene guarantees:
  1. The ground truth and the oracle never enter the workspace directory or the child's
     environment; the workspace holds only the white-box artifact (WB mode) and queries.json.
  2. The child process gets an allowlisted environment (no API keys or other host variables),
     runs with `python -E -s` (environment variables and user site-packages ignored; the
     workspace is on sys.path so `import artifact` works), under CPU-time and address-space
     limits (Linux `resource`) and a wall-clock timeout.
  3. MeteredOracle lives in the parent process, enforces max_queries exactly, accepts only
     vectors in {-1, +1}^n and appends every answered query to queries.json.

This is not a security boundary: the child runs as the same user, with the same file
system and network access as the harness.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .generators import BlackwellInvInstance, KXorInstance


def _make_rlimit_preexec(cpu_seconds: int, mem_mb: int) -> Callable[[], None]:
    def _apply() -> None:
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
            target = cpu_seconds if hard == resource.RLIM_INFINITY else min(cpu_seconds, hard)
            resource.setrlimit(resource.RLIMIT_CPU, (target, hard))
        except (ValueError, OSError):
            pass
        try:
            mem_bytes = int(mem_mb) * 1024 * 1024
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            target = mem_bytes if hard == resource.RLIM_INFINITY else min(mem_bytes, hard)
            resource.setrlimit(resource.RLIMIT_AS, (target, hard))
        except (ValueError, OSError):
            pass

    return _apply


class MeteredOracle:
    """Parent-process metered black-box oracle for KXorInstance or BlackwellInvInstance."""

    def __init__(
        self,
        instance: KXorInstance | BlackwellInvInstance,
        max_queries: int,
        queries_json_path: Optional[Path] = None,
    ) -> None:
        self._instance = instance
        self.n = int(instance.n)
        self.max_queries = int(max_queries)
        self.queries_used = 0
        self.history: List[Dict[str, Any]] = []
        self._queries_json_path = queries_json_path
        if self._queries_json_path is not None:
            self._persist()

    def _eval_batch(self, X: np.ndarray) -> np.ndarray:
        if isinstance(self._instance, KXorInstance):
            _, y = self._instance.eval_algebraic(X)
            return y
        return self._instance.eval_inv(X)

    def _persist(self) -> None:
        if self._queries_json_path is not None:
            payload = {
                "n": self.n,
                "max_queries": self.max_queries,
                "queries_used": self.queries_used,
                "remaining": self.max_queries - self.queries_used,
                "records": self.history,
            }
            self._queries_json_path.write_text(json.dumps(payload), encoding="utf-8")

    def query(self, inputs: Sequence[Sequence[int]]) -> Dict[str, Any]:
        """Evaluate a batch of vectors in {-1, +1}^n subject to the hard max_queries cap."""
        if not isinstance(inputs, (list, tuple)):
            return {
                "ok": False,
                "error": "inputs must be a list of vectors (or a single vector of length n)",
                "remaining": self.max_queries - self.queries_used,
            }
        # Normalize single vector [1, -1, ...] to batch [[1, -1, ...]]
        if len(inputs) > 0 and isinstance(inputs[0], (int, float)):
            batch_raw = [inputs]
        else:
            batch_raw = list(inputs)

        if len(batch_raw) == 0:
            return {
                "ok": True,
                "outputs": [],
                "queries_used": self.queries_used,
                "remaining": self.max_queries - self.queries_used,
            }

        remaining = self.max_queries - self.queries_used
        if remaining <= 0:
            return {
                "ok": False,
                "error": f"Query budget exhausted ({self.queries_used}/{self.max_queries}).",
                "outputs": [],
                "queries_used": self.queries_used,
                "remaining": 0,
            }

        truncated = False
        if len(batch_raw) > remaining:
            batch_raw = batch_raw[:remaining]
            truncated = True

        validated: List[List[int]] = []
        for idx, vec in enumerate(batch_raw):
            if not isinstance(vec, (list, tuple)) or len(vec) != self.n:
                return {
                    "ok": False,
                    "error": f"Vector #{idx} must have length {self.n}.",
                    "remaining": remaining,
                }
            row_int: List[int] = []
            for v in vec:
                if v not in (-1, 1, -1.0, 1.0):
                    return {
                        "ok": False,
                        "error": f"Vector #{idx} contains invalid entry {v!r}; entries must be -1 or +1.",
                        "remaining": remaining,
                    }
                row_int.append(int(v))
            validated.append(row_int)

        X = np.array(validated, dtype=np.int8)
        y = self._eval_batch(X)
        outputs = [int(v) for v in y.tolist()]

        for vec_int, out_int in zip(validated, outputs):
            self.history.append({"x": vec_int, "y": out_int})
        self.queries_used += len(outputs)
        self._persist()

        return {
            "ok": True,
            "outputs": outputs,
            "truncated_to_budget": truncated,
            "queries_used": self.queries_used,
            "remaining": self.max_queries - self.queries_used,
        }

    def query_random(self, count: int, seed: int = 0) -> Dict[str, Any]:
        """Convenience metered tool: evaluate `count` uniform random inputs in {-1, +1}^n."""
        if not isinstance(count, int) or count <= 0:
            return {
                "ok": False,
                "error": "count must be a positive integer.",
                "remaining": self.max_queries - self.queries_used,
            }
        remaining = self.max_queries - self.queries_used
        actual = min(count, remaining)
        if actual <= 0:
            return {
                "ok": False,
                "error": f"Query budget exhausted ({self.queries_used}/{self.max_queries}).",
                "outputs": [],
                "queries_used": self.queries_used,
                "remaining": 0,
            }
        rng_seed = int(seed) + 104729 * self.queries_used
        if rng_seed < 0:
            # The generator needs a non-negative seed. A negative `seed` that still gives one
            # (after earlier queries) is accepted, as it always was.
            return {
                "ok": False,
                "error": "seed must be a non-negative integer.",
                "remaining": self.max_queries - self.queries_used,
            }
        rng = np.random.default_rng(rng_seed)
        X = rng.choice([-1, 1], size=(actual, self.n)).astype(np.int8)
        res = self.query(X.tolist())
        # Include compact summary so tool-equipped agents can read full X/y from queries.json
        res["note"] = "All queried (x, y) pairs have been saved to queries.json in the workspace."
        return res


class IsolatedPythonSandbox:
    """Ephemeral sandbox workspace for tool-equipped LLM agents.

    Only contains white-box artifacts explicitly written by the harness (in WB mode) and
    `queries.json` written by the parent-process MeteredOracle. Never exposes host secrets.
    """

    def __init__(
        self,
        timeout_s: float = 10.0,
        mem_mb: int = 2048,
        max_output_chars: int = 8000,
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.mem_mb = int(mem_mb)
        self.max_output_chars = int(max_output_chars)
        self._tmpdir: Optional[tempfile.TemporaryDirectory[str]] = None
        self.workdir: Optional[Path] = None
        self.exec_count = 0
        self.total_wall_s = 0.0

    def __enter__(self) -> IsolatedPythonSandbox:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="vbb_pilot_sbx_")
        self.workdir = Path(self._tmpdir.name)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
            self.workdir = None

    @property
    def queries_json_path(self) -> Path:
        assert self.workdir is not None, "Sandbox not entered"
        return self.workdir / "queries.json"

    def write_text_artifact(self, filename: str, content: str) -> Path:
        assert self.workdir is not None, "Sandbox not entered"
        path = self.workdir / filename
        path.write_text(content, encoding="utf-8")
        return path

    def write_npz_artifact(self, filename: str, arrays: Dict[str, np.ndarray]) -> Path:
        assert self.workdir is not None, "Sandbox not entered"
        path = self.workdir / filename
        np.savez(path, **arrays)
        return path

    def run_python(self, code: str) -> Dict[str, Any]:
        """Execute user/agent Python code inside the isolated tempdir with scrubbed env."""
        assert self.workdir is not None, "Sandbox not entered"
        try:
            source = code.encode("utf-8")
        except UnicodeEncodeError:
            # A lone surrogate (from a JSON escape) cannot be written; the code is not run.
            return {"ok": False, "error": "the code is not valid UTF-8 text (it contains a lone surrogate); it was not run."}
        self.exec_count += 1
        script_path = self.workdir / f"_agent_step_{self.exec_count}.py"
        script_path.write_bytes(source)

        # Strict allowlisted environment — strips ANTHROPIC_API_KEY and all other host credentials
        clean_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": str(self.workdir),
            "TMPDIR": str(self.workdir),
        }

        cpu_limit = max(2, int(self.timeout_s) + 1)
        t0 = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                [sys.executable, "-E", "-s", str(script_path)],
                cwd=str(self.workdir),
                env=clean_env,
                capture_output=True,
                text=True,
                errors="replace",  # output that is not UTF-8 is returned with U+FFFD, not raised
                timeout=self.timeout_s,
                preexec_fn=_make_rlimit_preexec(cpu_limit, self.mem_mb),
            )
            returncode = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = -9
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = (
                ((exc.stderr or "") if isinstance(exc.stderr, str) else "")
                + f"\n[SANDBOX TIMEOUT after {self.timeout_s:.1f}s]"
            )
        wall_s = time.monotonic() - t0
        self.total_wall_s += wall_s

        def _trunc(s: str) -> str:
            if len(s) <= self.max_output_chars:
                return s
            half = self.max_output_chars // 2
            return s[:half] + f"\n... [truncated {len(s) - self.max_output_chars} chars] ...\n" + s[-half:]

        return {
            "ok": (returncode == 0) and (not timed_out),
            "returncode": returncode,
            "timed_out": timed_out,
            "wall_s": round(wall_s, 4),
            "stdout": _trunc(stdout),
            "stderr": _trunc(stderr),
        }
