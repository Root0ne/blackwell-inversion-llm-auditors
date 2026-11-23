"""Client selection on the command line: --provider, --base-url and the --client hook."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import List

from blackwell_inversion import cli
from blackwell_inversion.clients import AnthropicClient, GeminiClient, MockDryRunClient, OpenAICompatClient

from .test_openai_compat import hermetic_env

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "rule_based_client.py"


def _args(argv: List[str]):
    return cli.build_parser().parse_args(argv)


class TestProviderChoice(unittest.TestCase):
    def setUp(self) -> None:
        hermetic_env(self)

    def test_auto_keeps_the_original_choice(self) -> None:
        args = _args([])
        self.assertIsInstance(cli.make_client(args, ["gemini-3.8-flash", "gemma-4-31b-it"]), GeminiClient)
        self.assertIsInstance(cli.make_client(args, ["gemini-3.8-flash", "claude-haiku-5-5"]), AnthropicClient)
        self.assertIsInstance(cli.make_client(args, ["my-local-model"]), AnthropicClient)
        self.assertIsInstance(cli.make_client(_args(["--dry-run"]), ["x"]), MockDryRunClient)

    def test_explicit_providers(self) -> None:
        self.assertIsInstance(cli.make_client(_args(["--provider", "anthropic"]), ["gemini-3.8-flash"]), AnthropicClient)
        self.assertIsInstance(cli.make_client(_args(["--provider", "gemini"]), ["anything"]), GeminiClient)
        self.assertIsInstance(cli.make_client(_args(["--provider", "mock"]), ["x"]), MockDryRunClient)
        client = cli.make_client(_args([
            "--provider", "openai", "--base-url", "http://127.0.0.1:9/v1", "--max-cost-usd", "2",
            "--input-price-per-mtok", "0.5", "--output-price-per-mtok", "1.5", "--max-tokens-field", "max_completion_tokens",
            "--request-timeout-s", "30", "--api-key-env", "SOME_VAR"]), ["m"])
        self.assertIsInstance(client, OpenAICompatClient)
        self.assertEqual(
            (client.base_url, client.max_cost_usd, client.input_price_per_mtok, client.output_price_per_mtok,
             client.max_tokens_field, client.timeout_s, client.api_key_env, client.api_key),
            ("http://127.0.0.1:9/v1", 2.0, 0.5, 1.5, "max_completion_tokens", 30.0, "SOME_VAR", ""),
        )

    def test_provider_mock_runs_like_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            outs = []
            for flag in (["--provider", "mock"], ["--dry-run"]):
                out = Path(td) / f"{flag[-1]}.jsonl"
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.main(flag + ["--experiment", "exp2", "--only-bb", "--exp2-seeds", "1",
                                                      "--budget-tracks", "matched_total_budget", "--models", "x",
                                                      "--out", str(out)]), 0)
                rec = json.loads(out.read_text())
                rec.pop("wall_s")
                outs.append(rec)
            self.assertEqual(outs[0], outs[1])
            self.assertEqual(outs[0]["model"], "mock-x")

    def test_contradictory_options_are_rejected(self) -> None:
        for argv, message in (
            (["--provider", "openai"], "needs --base-url"),
            (["--base-url", "http://127.0.0.1:9/v1"], "only with --provider openai"),
            (["--dry-run", "--provider", "openai", "--base-url", "http://127.0.0.1:9/v1"], "cannot be combined"),
            (["--client", "examples.rule_based_client:make_client", "--provider", "anthropic"], "cannot be combined"),
            (["--client", "examples.rule_based_client:make_client", "--dry-run"], "cannot be combined"),
            (["--provider", "openai", "--base-url", "localhost:9"], "http(s) URL"),
            (["--provider", "openai", "--base-url", "http://127.0.0.1:9/v1", "--input-price-per-mtok", "-1"], "negative"),
        ):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as ctx:
                cli.main(argv + ["--out", "/nonexistent-dir-never-written/x.jsonl"])
            self.assertEqual(ctx.exception.code, 2)
            self.assertIn(message, err.getvalue(), argv)


class TestCustomClientHook(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def _fail(self, spec: str, message: str) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            cli.main(["--client", spec, "--out", str(self.dir / "never.jsonl")])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn(message, err.getvalue())
        self.assertFalse((self.dir / "never.jsonl").exists())

    def test_example_client_by_module_and_by_file(self) -> None:
        for spec in ("examples.rule_based_client:make_client", f"{EXAMPLE}:make_client"):
            out = self.dir / "toy.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["--client", spec, "--preset", "smoke", "--models", "toy", "--out", str(out)]), 0)
            records = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual([(r["mode"], r["variant"]) for r in records],
                             [("BB", "wb_inv_code"), ("WB", "wb_clean"), ("WB", "wb_inv_code")])
            for rec in records:
                self.assertEqual((rec["model"], rec["parse_ok"], rec["queries_used"], rec["cost_usd"]), ("toy", True, 80, 0.0))
                self.assertEqual(rec["exact_rule"], 1.0)  # 80 queries reveal a 3-variable majority rule
            out.unlink()

    def test_example_client_answers_exp1(self) -> None:
        out = self.dir / "toy1.jsonl"
        with contextlib.redirect_stdout(io.StringIO()):
            cli.main(["--client", "examples.rule_based_client:make_client", "--experiment", "exp1", "--regimes", "vbb_hard",
                      "--substrates", "code_clean", "--exp1-pairs", "1", "--models", "toy", "--out", str(out)])
        records = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual({r["decision_label"] for r in records}, {"CLEAN"})
        self.assertEqual(sum(r["correct"] for r in records), 2)  # right on the two clean instances

    def test_factory_receives_the_parsed_arguments(self) -> None:
        module = self.dir / "capture_client.py"
        module.write_text(
            "from blackwell_inversion.clients import MockDryRunClient\n"
            "SEEN = []\n"
            "def factory(args):\n"
            "    SEEN.append((args.models, args.max_cost_usd))\n"
            "    client = MockDryRunClient()\n"
            "    client.max_cost_usd = args.max_cost_usd\n"
            "    return client\n",
            encoding="utf-8",
        )
        args = _args(["--client", f"{module}:factory", "--models", "a,b", "--max-cost-usd", "7"])
        client = cli.make_client(args, ["a", "b"])
        self.assertEqual(client.max_cost_usd, 7.0)

    def test_bad_specs_fail_clearly(self) -> None:
        not_a_client = self.dir / "not_a_client.py"
        not_a_client.write_text("def factory(args):\n    return object()\nVALUE = 3\n", encoding="utf-8")
        self._fail("no_such_package.no_such_module:make", "cannot import 'no_such_package.no_such_module'")
        self._fail(str(self.dir / "missing.py") + ":make", "not found")
        self._fail("examples.rule_based_client:no_such_factory", "has no callable 'no_such_factory'")
        self._fail(f"{not_a_client}:VALUE", "has no callable 'VALUE'")
        self._fail(f"{not_a_client}:factory", "returned object, not a blackwell_inversion.clients.ModelClient")
        self._fail("examples.rule_based_client", "must have the form")
        self._fail(":make_client", "must have the form")


if __name__ == "__main__":
    unittest.main()
