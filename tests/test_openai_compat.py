"""The OpenAI-compatible client against a real HTTP server on 127.0.0.1.

Request bodies, both translation directions, every tolerance case listed in the module
docstring of `blackwell_inversion.clients.openai_compat`, retries, the spend cap and the
handling of the API key. Backoff sleeps are patched out; nothing leaves the machine.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import tempfile
import threading
import unittest
from typing import Any, Dict, List
from unittest import mock

from blackwell_inversion.agent import build_tool_schemas
from blackwell_inversion.clients import COST_KEY, BudgetExceededError, ModelClient, OpenAICompatClient
from blackwell_inversion.clients.openai_compat import RAW_MESSAGE_BLOCK, convert_messages, normalise_reply, to_openai_tools

from .fakes import LocalChatServer, chat_reply, function_call

USER = [{"role": "user", "content": "task"}]
SECRET = "sk-test-0123456789abcdef"
SLEEP = "blackwell_inversion.clients.openai_compat.time.sleep"


DEAD_PROXY = "http://127.0.0.1:9"


def hermetic_env(test: unittest.TestCase) -> None:
    """For one test: an empty HOME (so no ~/.env is read) and a proxy that refuses connections.

    No proxy exemption is set: requests to the loopback server succeed only because the
    client bypasses proxies for loopback hosts.
    """
    home = tempfile.TemporaryDirectory(prefix="oa_home_")
    test.addCleanup(home.cleanup)
    patcher = mock.patch.dict(os.environ, {"HOME": home.name, "http_proxy": DEAD_PROXY, "HTTP_PROXY": DEAD_PROXY})
    patcher.start()
    test.addCleanup(patcher.stop)
    for var in ("no_proxy", "NO_PROXY"):
        os.environ.pop(var, None)


class _ServerTest(unittest.TestCase):
    def setUp(self) -> None:
        hermetic_env(self)

    def serve(self, replies: List[Any]) -> LocalChatServer:
        queue = list(replies)
        server = LocalChatServer(lambda _i, _body, _headers: queue.pop(0))
        server.__enter__()
        self.addCleanup(server.__exit__, None, None, None)
        return server

    def call(self, server: LocalChatServer, client: OpenAICompatClient = None, **kwargs: Any) -> Dict[str, Any]:
        client = client or OpenAICompatClient(server.base_url)
        args = {"model": "local-model", "system": "SYS", "messages": USER, "max_tokens": 64}
        args.update(kwargs)
        return client.create_message(**args)


class TestRequests(_ServerTest):
    def test_request_body_and_headers(self) -> None:
        server = self.serve([chat_reply("hi")])
        tools = build_tool_schemas(True, 40)
        self.call(server, tools=tools, tool_choice={"type": "none"}, effort="medium", max_tokens=4096)
        req = server.requests[0]
        self.assertEqual(req["path"], "/v1/chat/completions")
        self.assertEqual(req["headers"]["content-type"], "application/json")
        self.assertNotIn("authorization", req["headers"])  # no key configured
        self.assertEqual(
            req["body"],
            {
                "model": "local-model",
                "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "task"}],
                "max_tokens": 4096,
                "tools": to_openai_tools(tools),
                "tool_choice": "none",
                "reasoning_effort": "medium",
            },
        )
        self.assertEqual(req["body"]["tools"][2]["function"]["parameters"], tools[2]["input_schema"])

    def test_optional_fields(self) -> None:
        server = self.serve([chat_reply("a"), chat_reply("b")])
        client = OpenAICompatClient(server.base_url + "/", max_tokens_field="max_completion_tokens")
        self.call(server, client, effort="")  # no effort: no reasoning_effort; no tools: no tool_choice
        self.call(server, client, tools=build_tool_schemas(False, 40), tool_choice=None, effort="high",
                  thinking_mode="disabled", thinking_budget=5000)
        first, second = server.bodies()
        self.assertEqual(set(first), {"model", "messages", "max_completion_tokens"})
        self.assertNotIn("tool_choice", second)  # tools allowed: the field is omitted
        self.assertEqual(second["reasoning_effort"], "high")
        self.assertNotIn("thinking", json.dumps(second))  # thinking settings are not sent

    def test_bearer_key_from_the_named_variable(self) -> None:
        server = self.serve([chat_reply("ok")])
        with mock.patch.dict(os.environ, {"MY_LLM_KEY": SECRET}):
            client = OpenAICompatClient(server.base_url, api_key_env="MY_LLM_KEY")
        self.call(server, client)
        self.assertEqual(server.requests[0]["headers"]["authorization"], f"Bearer {SECRET}")

    def test_constructor_validation(self) -> None:
        for bad in ("", "127.0.0.1:8000/v1", "ftp://host/v1", "http://"):
            with self.assertRaises(ValueError):
                OpenAICompatClient(bad)
        with self.assertRaises(ValueError):
            OpenAICompatClient("http://127.0.0.1:1/v1", max_tokens_field="max_output_tokens")
        with self.assertRaises(ValueError):
            OpenAICompatClient("http://127.0.0.1:1/v1", input_price_per_mtok=-1.0)
        self.assertTrue(issubclass(OpenAICompatClient, ModelClient))


class TestTranslation(unittest.TestCase):
    def test_conversation_to_chat_messages(self) -> None:
        messages = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "thinking aloud"},
                {"type": "tool_use", "id": "a1", "name": "query_oracle_random", "input": {"count": 5, "seed": 1}},
                {"type": "tool_use", "id": "a2", "name": "run_python", "input": "{broken"},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "a1", "content": '{"ok": true}'},
                {"type": "tool_result", "tool_use_id": "a2", "content": [{"type": "text", "text": "err"}]},
                {"type": "text", "text": "note 1"},
                {"type": "text", "text": "note 2"},
            ]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "a3", "name": "query_oracle", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a3", "content": "{}"}]},
            {"role": "assistant", "content": []},
            {"role": "assistant", "content": "plain"},
        ]
        out = convert_messages("SYS", messages)
        self.assertEqual(out, [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "thinking aloud", "tool_calls": [
                {"id": "a1", "type": "function", "function": {"name": "query_oracle_random", "arguments": '{"count": 5, "seed": 1}'}},
                {"id": "a2", "type": "function", "function": {"name": "run_python", "arguments": "{broken"}},
            ]},
            {"role": "tool", "tool_call_id": "a1", "content": '{"ok": true}'},
            {"role": "tool", "tool_call_id": "a2", "content": "err"},
            {"role": "user", "content": "note 1\n\nnote 2"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "a3", "type": "function", "function": {"name": "query_oracle", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "a3", "content": "{}"},
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "plain"},
        ])
        self.assertEqual(convert_messages("", USER), [{"role": "user", "content": "task"}])

    def test_reply_to_contract(self) -> None:
        raw = chat_reply("I will query.", [function_call("query_oracle_random", '{"count": 3}', "x9")],
                         usage={"prompt_tokens": 120, "completion_tokens": 30,
                                "completion_tokens_details": {"reasoning_tokens": 12}})
        self.assertEqual(normalise_reply(raw, n_messages=1), {
            "content": [{"type": RAW_MESSAGE_BLOCK, "message": raw["choices"][0]["message"]},
                        {"type": "text", "text": "I will query."},
                        {"type": "tool_use", "id": "x9", "name": "query_oracle_random", "input": {"count": 3}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 120, "output_tokens": 30, "output_tokens_details": {"thinking_tokens": 12}},
        })

    def test_round_trip_replays_the_model_turn(self) -> None:
        raw = chat_reply(None, [function_call("run_python", '{"code": "print(1)"}', "r1"),
                                function_call("query_oracle_random", "not json", "r2")])
        turn = normalise_reply(raw, n_messages=1)["content"]
        replay = convert_messages("", [{"role": "assistant", "content": turn}])[0]
        self.assertEqual(replay, {"role": "assistant", "content": None, "tool_calls": raw["choices"][0]["message"]["tool_calls"]})


def _replay(raw: Dict[str, Any], n_messages: int = 1) -> Dict[str, Any]:
    """The assistant message sent back on the next request for one reply."""
    turn = normalise_reply(raw, n_messages=n_messages)["content"]
    return convert_messages("", [{"role": "assistant", "content": turn}])[0]


class TestReplayOfServerFields(unittest.TestCase):
    """The server's assistant message is sent back as it came; only ids and arguments are normalised."""

    def test_extra_content_and_reasoning_fields_are_sent_back(self) -> None:
        sig = {"google": {"thought_signature": "opaque=="}}
        details = [{"type": "reasoning.encrypted", "data": "xyz"}]
        raw = chat_reply("Let me look.", [function_call("query_oracle_random", '{"count": 5}', "a") | {"extra_content": sig}],
                         reasoning_content="step 1", reasoning="step 1", reasoning_details=details)
        message = raw["choices"][0]["message"]
        replay = _replay(raw)
        self.assertEqual(replay, message)
        self.assertIsNot(replay, message)
        self.assertEqual(replay["tool_calls"][0]["extra_content"], sig)
        # The visible turn is unchanged: the reasoning is not part of the answer text.
        self.assertEqual([b for b in normalise_reply(raw, n_messages=1)["content"] if b["type"] != RAW_MESSAGE_BLOCK],
                         [{"type": "text", "text": "Let me look."},
                          {"type": "tool_use", "id": "a", "name": "query_oracle_random", "input": {"count": 5}}])

    def test_synthesised_ids_replace_the_server_ids(self) -> None:
        calls = [function_call("query_oracle", "{}", None) | {"extra_content": {"k": 1}},
                 function_call("query_oracle", "{}", "dup"), function_call("query_oracle", "{}", "dup")]
        replay = _replay(chat_reply(None, calls), n_messages=3)
        self.assertEqual([c["id"] for c in replay["tool_calls"]], ["call_3_0", "dup", "call_3_2"])
        self.assertEqual(replay["tool_calls"][0]["extra_content"], {"k": 1})

    def test_arguments_are_sent_back_as_text(self) -> None:
        cases = (
            ('{"count":5}', '{"count":5}'),  # the server's own text, byte for byte
            ("{oops", "{oops"),
            ({"count": 4}, '{"count": 4}'),  # an object (non-standard) becomes its JSON text
            (None, "{}"),
            (7, "7"),
        )
        for sent, replayed in cases:
            replay = _replay(chat_reply(None, [function_call("query_oracle_random", sent, "x")]))
            self.assertEqual(replay["tool_calls"][0]["function"]["arguments"], replayed, sent)

    def test_plain_reply_replays_as_before(self) -> None:
        self.assertEqual(_replay(chat_reply("only text")), {"role": "assistant", "content": "only text"})
        self.assertEqual(_replay(chat_reply(None)), {"role": "assistant", "content": None})
        with_empty_calls = chat_reply("x")
        with_empty_calls["choices"][0]["message"]["tool_calls"] = []
        self.assertEqual(_replay(with_empty_calls), {"role": "assistant", "content": "x"})  # [] is not sent back
        no_role = chat_reply("y")
        del no_role["choices"][0]["message"]["role"]
        self.assertEqual(_replay(no_role), {"role": "assistant", "content": "y"})

    def test_turns_without_the_raw_message_are_rebuilt(self) -> None:
        # A conversation built by other code (or an older version) has no raw-message block.
        turn = [{"type": "text", "text": "t"}, {"type": "tool_use", "id": "i", "name": "query_oracle", "input": {}}]
        self.assertEqual(convert_messages("", [{"role": "assistant", "content": turn}])[0], {
            "role": "assistant", "content": "t",
            "tool_calls": [{"id": "i", "type": "function", "function": {"name": "query_oracle", "arguments": "{}"}}]})


class TestTolerance(unittest.TestCase):
    """How replies that real servers send are read (module docstring of openai_compat)."""

    def norm(self, raw: Dict[str, Any], n_messages: int = 3) -> Dict[str, Any]:
        """The normalised reply without its leading raw-message block (checked here)."""
        res = normalise_reply(raw, n_messages=n_messages)
        self.assertEqual(res["content"][0], {"type": RAW_MESSAGE_BLOCK, "message": raw["choices"][0]["message"]})
        res["content"] = res["content"][1:]
        return res

    def test_missing_or_malformed_usage_counts_as_zero(self) -> None:
        zero = {"input_tokens": 0, "output_tokens": 0, "output_tokens_details": {"thinking_tokens": 0}}
        self.assertEqual(self.norm(chat_reply("x"))["usage"], zero)
        self.assertEqual(self.norm(chat_reply("x", usage={"prompt_tokens": None, "completion_tokens": "7",
                                                          "completion_tokens_details": None}))["usage"], zero)
        negative = {"prompt_tokens": -5, "completion_tokens": -1.5, "completion_tokens_details": {"reasoning_tokens": -2}}
        self.assertEqual(self.norm(chat_reply("x", usage=negative))["usage"], zero)

    def test_null_content_and_text_parts(self) -> None:
        self.assertEqual(self.norm(chat_reply(None))["content"], [])
        parts = [{"type": "text", "text": "ab"}, {"type": "text", "text": "c"}]
        self.assertEqual(self.norm(chat_reply(parts))["content"], [{"type": "text", "text": "abc"}])

    def test_arguments_that_are_not_a_json_object_stay_raw(self) -> None:
        for raw_args, expected in (
            ("{oops", "{oops"),
            ("[1, 2]", "[1, 2]"),
            ('"count"', '"count"'),
            ("", {}),
            ("   ", {}),
            (None, {}),
            ({"count": 4}, {"count": 4}),  # already decoded by the server
            ('{"count": "five"}', {"count": "five"}),  # a JSON object; the agent loop checks the values
            ('{"count": ' + "9" * 5000 + "}", '{"count": ' + "9" * 5000 + "}"),  # an integer too long to convert
        ):
            got = self.norm(chat_reply(None, [function_call("query_oracle_random", raw_args)]))
            self.assertEqual(got["content"][0]["input"], expected, raw_args)

    def test_missing_and_repeated_ids_are_synthesised(self) -> None:
        calls = [function_call("query_oracle", "{}", None), function_call("query_oracle", "{}", "dup"),
                 function_call("query_oracle", "{}", "dup"), function_call("query_oracle", "{}", "")]
        ids = [b["id"] for b in self.norm(chat_reply(None, calls), n_messages=5)["content"]]
        self.assertEqual(ids, ["call_5_0", "dup", "call_5_2", "call_5_3"])
        # Stable: the same reply at the same point of the conversation gets the same ids.
        self.assertEqual(ids, [b["id"] for b in self.norm(chat_reply(None, calls), n_messages=5)["content"]])

    def test_finish_reason_mapping(self) -> None:
        call = [function_call("query_oracle", "{}")]
        cases = [
            (chat_reply("x", finish_reason="stop"), "end_turn"),
            (chat_reply("x", finish_reason="length"), "max_tokens"),
            (chat_reply("x", finish_reason=None), "end_turn"),
            (chat_reply(None, call, finish_reason="tool_calls"), "tool_use"),
            (chat_reply(None, call, finish_reason="stop"), "tool_use"),  # tool calls win over "stop"
            (chat_reply(None, call, finish_reason="length"), "max_tokens"),  # cut at the cap
        ]
        cases[2][0]["choices"][0]["finish_reason"] = None
        for raw, expected in cases:
            self.assertEqual(self.norm(raw)["stop_reason"], expected)

    def test_reasoning_text_is_not_part_of_the_turn(self) -> None:
        # It stays in the raw-message block only, to be sent back (TestReplayOfServerFields).
        for field in ("reasoning_content", "reasoning"):
            raw = chat_reply('{"support": [1, 2, 3], "signs": [1, 1, 1]}', **{field: '{"support": [7, 8, 9]} hmm'})
            got = self.norm(raw)
            self.assertEqual(got["content"], [{"type": "text", "text": '{"support": [1, 2, 3], "signs": [1, 1, 1]}'}])
            self.assertNotIn("hmm", json.dumps(got))
        self.assertEqual(self.norm(chat_reply(None, reasoning_content="only thoughts"))["content"], [])

    def test_uninterpretable_replies_raise(self) -> None:
        bad_call = chat_reply(None, [{"type": "function", "function": {"arguments": "{}"}}])
        no_message = chat_reply("x")
        del no_message["choices"][0]["message"]
        bad_calls = chat_reply("x")
        bad_calls["choices"][0]["message"]["tool_calls"] = {"name": "x"}
        for raw in ([], {"error": {"message": "model not found"}}, {"choices": []}, {"choices": ["x"]},
                    no_message, bad_calls, bad_call, chat_reply({"text": "dict content"})):
            with self.assertRaises(RuntimeError, msg=json.dumps(raw)[:80]):
                self.norm(raw)


class TestHTTPBehaviour(_ServerTest):
    def test_end_to_end_reply_and_cost(self) -> None:
        reply = chat_reply("done", usage={"prompt_tokens": 1_000_000, "completion_tokens": 500_000})
        server = self.serve([reply])
        client = OpenAICompatClient(server.base_url, input_price_per_mtok=0.2, output_price_per_mtok=1.0)
        resp = self.call(server, client)
        self.assertEqual(resp["content"][1:], [{"type": "text", "text": "done"}])
        self.assertAlmostEqual(resp[COST_KEY], 0.7)
        self.assertAlmostEqual(client.total_cost_usd, 0.7)
        self.assertEqual((client.total_input_tokens, client.total_output_tokens), (1_000_000, 500_000))

    def test_local_model_costs_nothing(self) -> None:
        server = self.serve([chat_reply("x", usage={"prompt_tokens": 10**9, "completion_tokens": 10**9})])
        client = OpenAICompatClient(server.base_url, max_cost_usd=0.0001)
        self.assertEqual(self.call(server, client)[COST_KEY], 0.0)

    def test_retries_on_429_and_5xx_then_succeeds(self) -> None:
        server = self.serve([(429, "{}", {"retry-after": "3"}), (503, "busy", {}), (500, "{}", {}), chat_reply("ok")])
        with mock.patch(SLEEP) as sleep:
            resp = self.call(server)
        self.assertEqual(resp["stop_reason"], "end_turn")
        self.assertEqual(len(server.requests), 4)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [3.0, 2.0, 4.0])
        self.assertEqual(len({json.dumps(b) for b in server.bodies()}), 1)  # the same request each time

    def test_gives_up_after_max_retries(self) -> None:
        server = self.serve([(502, "bad gateway", {})] * 3)
        client = OpenAICompatClient(server.base_url, max_retries=3)
        with mock.patch(SLEEP), self.assertRaises(RuntimeError) as ctx:
            self.call(server, client)
        self.assertEqual(len(server.requests), 3)
        self.assertIn("HTTP 502", str(ctx.exception))
        self.assertIn("bad gateway", str(ctx.exception))

    def test_client_errors_are_not_retried(self) -> None:
        server = self.serve([(400, '{"error": "bad request"}', {})])
        with mock.patch(SLEEP) as sleep, self.assertRaises(RuntimeError) as ctx:
            self.call(server)
        self.assertEqual(len(server.requests), 1)
        sleep.assert_not_called()
        self.assertIn("HTTP 400", str(ctx.exception))

    def test_payment_required_stops_like_the_spend_cap(self) -> None:
        server = self.serve([(402, "no credit", {})])
        with self.assertRaises(BudgetExceededError):
            self.call(server)

    def test_non_json_reply_raises(self) -> None:
        server = self.serve([(200, "<html>proxy login</html>", {})])
        with self.assertRaises(RuntimeError) as ctx:
            self.call(server)
        self.assertIn("not JSON", str(ctx.exception))

    def test_network_errors_are_retried_then_reported(self) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        client = OpenAICompatClient(f"http://127.0.0.1:{port}/v1", max_retries=2)
        with mock.patch(SLEEP) as sleep, self.assertRaises(RuntimeError) as ctx:
            client.create_message(model="m", system="", messages=USER, max_tokens=8)
        message = str(ctx.exception)
        self.assertIn(f"cannot reach the OpenAI-compatible server at http://127.0.0.1:{port}/v1", message)
        self.assertIn("is it running", message)
        self.assertNotIn("\n", message)
        self.assertEqual(sleep.call_count, 1)

    def test_read_timeout_is_not_retried(self) -> None:
        release = threading.Event()

        def stalls(_i: int, _body: Any, _headers: Dict[str, str]) -> Any:
            release.wait(30)  # bounded: released by the test as soon as the client has given up
            return chat_reply("late")

        server = LocalChatServer(stalls)
        server.__enter__()
        self.addCleanup(server.__exit__, None, None, None)
        self.addCleanup(release.set)  # runs first, so the handler thread ends before the server closes
        client = OpenAICompatClient(server.base_url, timeout_s=0.2)
        with mock.patch(SLEEP) as sleep, self.assertRaises(RuntimeError) as ctx:
            self.call(server, client)
        release.set()
        self.assertEqual(len(server.requests), 1)  # one generation, not five
        sleep.assert_not_called()
        self.assertIn("no reply within 0.2 s", str(ctx.exception))
        self.assertIn("--request-timeout-s", str(ctx.exception))

    def test_loopback_requests_bypass_the_proxy(self) -> None:
        # hermetic_env points http_proxy at a closed port and sets no exemption.
        self.assertEqual(os.environ["http_proxy"], DEAD_PROXY)
        self.assertNotIn("no_proxy", os.environ)
        for host in ("127.0.0.1", "localhost"):
            server = self.serve([chat_reply("direct")])
            client = OpenAICompatClient(server.base_url.replace("127.0.0.1", host))
            self.assertEqual(self.call(server, client)["content"][1:], [{"type": "text", "text": "direct"}])

    def test_warning_for_a_key_sent_without_tls(self) -> None:
        cases = (
            ("http://203.0.113.7:8000/v1", "MY_LLM_KEY", True),
            ("https://203.0.113.7/v1", "MY_LLM_KEY", False),
            ("http://127.0.0.1:8000/v1", "MY_LLM_KEY", False),
            ("http://localhost:8000/v1", "MY_LLM_KEY", False),
            ("http://[::1]:8000/v1", "MY_LLM_KEY", False),
            ("http://203.0.113.7:8000/v1", "", False),
        )
        for url, key_env, warned in cases:
            err = io.StringIO()
            with mock.patch.dict(os.environ, {"MY_LLM_KEY": SECRET}), contextlib.redirect_stderr(err):
                OpenAICompatClient(url, api_key_env=key_env)
            self.assertEqual("[WARNING]" in err.getvalue(), warned, url)
            self.assertNotIn(SECRET, err.getvalue())
            if warned:
                self.assertIn("203.0.113.7", err.getvalue())
                self.assertEqual(err.getvalue().count("\n"), 1)

    def test_spend_cap_refuses_the_next_request(self) -> None:
        reply = chat_reply("x", usage={"prompt_tokens": 1_000_000, "completion_tokens": 0})
        server = self.serve([reply, reply, reply])
        client = OpenAICompatClient(server.base_url, max_cost_usd=1.5, input_price_per_mtok=1.0)
        self.call(server, client)
        self.call(server, client)
        with self.assertRaises(BudgetExceededError):
            self.call(server, client)
        self.assertEqual(len(server.requests), 2)
        self.assertAlmostEqual(client.total_cost_usd, 2.0)

    def test_missing_key_fails_when_the_client_is_made(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            OpenAICompatClient("http://127.0.0.1:9/v1", api_key_env="UNSET_TEST_KEY_VAR")
        self.assertIn("UNSET_TEST_KEY_VAR", str(ctx.exception))
        self.assertIn("--api-key-env", str(ctx.exception))

    def test_key_from_the_home_env_file(self) -> None:
        server = self.serve([chat_reply("x")])
        with open(os.path.join(os.environ["HOME"], ".env"), "w", encoding="utf-8") as fh:
            fh.write(f"FILE_KEY_VAR='{SECRET}'\n")
        self.call(server, OpenAICompatClient(server.base_url, api_key_env="FILE_KEY_VAR"))
        self.assertEqual(server.requests[0]["headers"]["authorization"], f"Bearer {SECRET}")

    def test_key_never_appears_in_exceptions(self) -> None:
        def echo_auth(_i: int, _body: Any, headers: Dict[str, str]) -> Any:
            return (401, json.dumps({"error": f"invalid key {headers.get('authorization')}"}), {})

        server = LocalChatServer(echo_auth)
        with server, mock.patch.dict(os.environ, {"MY_LLM_KEY": SECRET}):
            client = OpenAICompatClient(server.base_url, api_key_env="MY_LLM_KEY")
            with self.assertRaises(RuntimeError) as ctx:
                client.create_message(model="m", system="", messages=USER, max_tokens=8)
        exc = ctx.exception
        texts = [str(exc), repr(exc), str(exc.__cause__), repr(exc.__cause__)]
        self.assertTrue(all(SECRET not in t for t in texts))
        self.assertIn("[REDACTED]", str(exc))
        self.assertIn("HTTP 401", str(exc))


if __name__ == "__main__":
    unittest.main()
