"""Provider clients: the ModelClient contract, cost accounting, the spend cap and request translation."""

from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest import mock

from blackwell_inversion.agent import build_tool_schemas
from blackwell_inversion.clients import (
    COST_KEY,
    AnthropicClient,
    BudgetExceededError,
    GeminiClient,
    MockDryRunClient,
    ModelClient,
    estimate_cost_usd,
)
from blackwell_inversion.clients.anthropic import apply_prompt_caching
from blackwell_inversion.clients.gemini import convert_messages, to_gemini_schema
from blackwell_inversion.clients.pricing import MODEL_PRICING_USD_PER_MTOK

from .fakes import FakeHTTP

USER = [{"role": "user", "content": "hi"}]
ANTHROPIC_OK = {
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1_000_000, "output_tokens": 0, "cache_read_input_tokens": 1_000_000},
}


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.invalid", code, "error", {}, io.BytesIO(b"{}"))


class TestPricingAndCaching(unittest.TestCase):
    def test_prompt_caching_and_cost_accounting(self) -> None:
        msgs = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "{}"}]},
        ]
        tools = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
        sys_c, tools_c, msgs_c = apply_prompt_caching("sys", tools, msgs)
        self.assertEqual(sys_c[0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(tools_c[-1]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(msgs_c[-1]["content"][-1]["cache_control"], {"type": "ephemeral"})
        # The caller's objects are untouched.
        self.assertNotIn("cache_control", tools[-1])
        self.assertNotIn("cache_control", msgs[-1]["content"][-1])
        c_full = estimate_cost_usd("claude-sonnet-5-5", 1_000_000, 0)
        c_read = estimate_cost_usd("claude-sonnet-5-5", 0, 0, cache_read_tokens=1_000_000)
        self.assertAlmostEqual(c_full, 2.0, places=6)
        self.assertAlmostEqual(c_read, 0.2, places=6)

    def test_every_pricing_entry_is_reachable(self) -> None:
        # The first matching prefix wins, so an earlier prefix of a later entry would shadow it.
        prefixes = [p for p, _, _ in MODEL_PRICING_USD_PER_MTOK]
        for j, later in enumerate(prefixes):
            for earlier in prefixes[:j]:
                self.assertFalse(later.startswith(earlier), f"{earlier!r} shadows {later!r}")

    def test_unknown_models_use_the_conservative_fallback(self) -> None:
        self.assertAlmostEqual(estimate_cost_usd("some-new-model", 1_000_000, 1_000_000), 30.0)
        self.assertEqual(estimate_cost_usd("mock-claude-haiku-5-5", 10**9, 10**9), 0.0)
        self.assertAlmostEqual(estimate_cost_usd("claude-haiku-5-5", 0, 0, cache_write_tokens=1_000_000), 0.125)


class TestModelClientContract(unittest.TestCase):
    def test_abstract_base_requires_create_message(self) -> None:
        with self.assertRaises(TypeError):
            ModelClient(max_cost_usd=1.0)  # type: ignore[abstract]
        for cls in (AnthropicClient, GeminiClient, MockDryRunClient):
            self.assertTrue(issubclass(cls, ModelClient))

    def test_budget_guard_raises_before_http(self) -> None:
        client = AnthropicClient(api_key="dummy-never-sent", max_cost_usd=1.0)
        client.total_cost_usd = 1.01
        with self.assertRaises(BudgetExceededError):
            client.create_message(
                model="claude-haiku-5-5",
                system="s",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=16,
            )

    def test_spend_accumulates_until_the_cap_stops_further_requests(self) -> None:
        fake = FakeHTTP([ANTHROPIC_OK, ANTHROPIC_OK, ANTHROPIC_OK])
        client = AnthropicClient(api_key="k", max_cost_usd=0.15)
        with fake.patch():
            first = client.create_message(model="claude-haiku-5-5", system="s", messages=USER, max_tokens=8)
            self.assertAlmostEqual(first[COST_KEY], 0.11)  # 1M input at 0.10 + 1M cache reads at 0.01
            client.create_message(model="claude-haiku-5-5", system="s", messages=USER, max_tokens=8)
            with self.assertRaises(BudgetExceededError):
                client.create_message(model="claude-haiku-5-5", system="s", messages=USER, max_tokens=8)
        self.assertEqual(len(fake.requests), 2)
        self.assertAlmostEqual(client.total_cost_usd, 0.22)
        self.assertEqual(client.total_input_tokens, 2_000_000)
        self.assertEqual(client.total_cache_read_tokens, 2_000_000)

    def test_missing_api_key_is_reported_without_a_request(self) -> None:
        client = AnthropicClient(api_key="placeholder", max_cost_usd=1.0)
        client.api_key = ""  # as if neither the environment nor ~/.env held a key
        with self.assertRaises(RuntimeError):
            client.create_message(model="claude-haiku-5-5", system="s", messages=USER, max_tokens=8)


class TestAnthropicRequests(unittest.TestCase):
    def _body(self, **kwargs: object) -> dict:
        args = {
            "model": "claude-haiku-5-5",
            "system": "s",
            "messages": USER,
            "max_tokens": 4096,
            "tools": build_tool_schemas(True, 40),
            "tool_choice": None,
            "thinking_budget": 0,
            "thinking_mode": "adaptive",
            "effort": "medium",
        }
        args.update(kwargs)
        return AnthropicClient(api_key="k").build_request_body(**args)  # type: ignore[arg-type]

    def test_thinking_effort_and_temperature(self) -> None:
        self.assertEqual(self._body()["thinking"], {"type": "adaptive"})
        self.assertEqual(self._body()["output_config"], {"effort": "medium"})
        self.assertEqual(self._body(thinking_mode="disabled")["thinking"], {"type": "disabled"})
        budget = self._body(thinking_mode="budget", thinking_budget=5000, max_tokens=4096)
        self.assertEqual(budget["thinking"], {"type": "enabled", "budget_tokens": 5000})
        self.assertEqual(budget["max_tokens"], 6024)  # raised above the thinking budget
        plain = self._body(model="claude-sonnet-4-5-20250929", thinking_mode="default", effort="")
        self.assertNotIn("thinking", plain)
        self.assertNotIn("output_config", plain)
        self.assertEqual(plain["temperature"], 0.0)
        self.assertNotIn("temperature", self._body(model="claude-sonnet-5-5", thinking_mode="default"))

    def test_tool_choice_is_sent_only_with_tools(self) -> None:
        self.assertEqual(self._body(tool_choice={"type": "none"})["tool_choice"], {"type": "none"})
        self.assertNotIn("tool_choice", self._body(tools=None, tool_choice={"type": "none"}))

    def test_retries_on_overload_then_succeeds(self) -> None:
        fake = FakeHTTP([http_error(529), ANTHROPIC_OK])
        client = AnthropicClient(api_key="k", max_cost_usd=10.0)
        with fake.patch(), mock.patch("blackwell_inversion.clients.anthropic.time.sleep") as sleep:
            resp = client.create_message(model="claude-haiku-5-5", system="s", messages=USER, max_tokens=8)
        self.assertEqual(resp["stop_reason"], "end_turn")
        self.assertEqual(len(fake.requests), 2)
        sleep.assert_called_once_with(1.0)
        self.assertEqual(fake.requests[0]["headers"]["x-api-key"], "k")
        self.assertEqual(fake.requests[0]["url"], "https://api.anthropic.com/v1/messages")

    def test_client_errors_are_not_retried(self) -> None:
        fake = FakeHTTP([http_error(400)])
        with fake.patch(), self.assertRaises(RuntimeError):
            AnthropicClient(api_key="k").create_message(model="claude-haiku-5-5", system="s", messages=USER, max_tokens=8)
        self.assertEqual(len(fake.requests), 1)


class TestGeminiTranslation(unittest.TestCase):
    def test_schema_translation(self) -> None:
        schema = build_tool_schemas(True, 40)[1]["input_schema"]
        out = to_gemini_schema(schema)
        self.assertEqual(out["type"], "OBJECT")
        self.assertEqual(out["properties"]["inputs"]["items"]["items"]["type"], "INTEGER")
        self.assertNotIn("enum", out["properties"]["inputs"]["items"]["items"])  # enums only on strings
        self.assertEqual(out["required"], ["inputs"])

    def test_messages_tool_results_and_raw_parts(self) -> None:
        raw = [{"functionCall": {"name": "query_oracle", "args": {}}, "thoughtSignature": "sig"}]
        messages = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": [{"type": "_gemini_raw_parts", "parts": raw},
                                              {"type": "tool_use", "id": "g1", "name": "query_oracle", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "g1", "content": json.dumps([1, 0])},
                                         {"type": "text", "text": "note"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ]
        contents = convert_messages(messages, inline_system="SYS")
        self.assertEqual(contents[0], {"role": "user", "parts": [{"text": "[System Instructions]\nSYS\n\n"}, {"text": "task"}]})
        self.assertEqual(contents[1], {"role": "model", "parts": raw})  # replayed verbatim
        self.assertEqual(
            contents[2]["parts"][0],
            {"functionResponse": {"name": "query_oracle", "response": {"result": [1, 0]}}},
        )
        self.assertEqual(contents[3], {"role": "model", "parts": [{"text": "answer"}]})

    def test_response_normalisation_and_costs(self) -> None:
        reply = {
            "candidates": [{"content": {"parts": [{"text": "thinking", "thought": True},
                                                  {"functionCall": {"name": "run_python", "args": {"code": "1"}}}]},
                            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 1_000_000, "cachedContentTokenCount": 400_000,
                              "candidatesTokenCount": 100_000, "thoughtsTokenCount": 100_000},
        }
        fake = FakeHTTP([reply])
        client = GeminiClient(api_key="k", max_cost_usd=10.0)
        with fake.patch():
            resp = client.create_message(model="gemini-3.5-flash-lite", system="S", messages=USER, max_tokens=64,
                                         tools=build_tool_schemas(True, 40), tool_choice={"type": "none"}, effort="high")
        body = fake.requests[0]["body"]
        self.assertEqual(body["systemInstruction"], {"parts": [{"text": "S"}]})
        self.assertEqual(body["generationConfig"], {"maxOutputTokens": 64, "thinkingConfig": {"thinkingLevel": "high"}})
        self.assertEqual(body["toolConfig"], {"functionCallingConfig": {"mode": "NONE"}})
        self.assertEqual(resp["stop_reason"], "tool_use")
        self.assertEqual([b["type"] for b in resp["content"]], ["_gemini_raw_parts", "tool_use"])
        self.assertEqual(resp["usage"]["output_tokens"], 200_000)
        self.assertEqual(resp["usage"]["output_tokens_details"], {"thinking_tokens": 100_000})
        # 600k uncached input at 0.30 + 400k cached at 0.03 + 200k output at 2.50 per million.
        self.assertAlmostEqual(resp[COST_KEY], 0.18 + 0.012 + 0.5)
        self.assertAlmostEqual(client.total_cost_usd, resp[COST_KEY])

    def test_gemma_inlines_the_system_prompt(self) -> None:
        fake = FakeHTTP([{"candidates": [{"content": {"parts": [{"text": "x"}]}, "finishReason": "MAX_TOKENS"}]}])
        with fake.patch():
            resp = GeminiClient(api_key="k").create_message(model="gemma-4-31b-it", system="S", messages=USER, max_tokens=8)
        body = fake.requests[0]["body"]
        self.assertNotIn("systemInstruction", body)
        self.assertEqual(body["generationConfig"], {"maxOutputTokens": 8})
        self.assertTrue(body["contents"][0]["parts"][0]["text"].startswith("[System Instructions]"))
        self.assertEqual(resp["stop_reason"], "max_tokens")

    def test_payment_required_stops_like_the_spend_cap(self) -> None:
        fake = FakeHTTP([http_error(402)])
        with fake.patch(), self.assertRaises(BudgetExceededError):
            GeminiClient(api_key="k").create_message(model="gemini-3.8-flash", system="S", messages=USER, max_tokens=8)


class TestMockClient(unittest.TestCase):
    def test_mock_rejects_non_alternating_roles(self) -> None:
        with self.assertRaises(AssertionError):
            MockDryRunClient().create_message(
                model="mock-x", system="s", messages=[{"role": "assistant", "content": "x"}], max_tokens=8
            )

    def test_mock_never_spends(self) -> None:
        client = MockDryRunClient()
        resp = client.create_message(model="mock-x", system="s", messages=[{"role": "user", "content": "task"}],
                                     max_tokens=8, tools=build_tool_schemas(False, 40))
        self.assertEqual(resp[COST_KEY], 0.0)
        self.assertEqual(client.total_cost_usd, 0.0)


if __name__ == "__main__":
    unittest.main()
