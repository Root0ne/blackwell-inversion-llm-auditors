# Evaluating your own model

This guide runs the benchmark on a model of your choice and puts its result next to the
paper's. It covers a local model behind an OpenAI-compatible server, a hosted
OpenAI-compatible API, and a client of your own. Everything below runs from the
repository root after the setup in the [README](../README.md#quick-start). Use the
repository from a checkout (`pip install -e .` or `pip install -r requirements.txt`): the
paper's report that `--compare-paper` reads and the example client live in the checkout,
not in an installed package.

## 1. What the model must support

- **Tool calling** in the OpenAI function-calling format (`tools`, `tool_calls`, `role:
  "tool"` messages). The agent works only through three tools: `query_oracle_random`,
  `query_oracle` and `run_python`. A model that never calls a tool still produces records,
  but it can only guess.
- **Context.** A trial is at most five requests in Exp 2 and six in Exp 1, and each request
  repeats the whole conversation. In the paper's logs the prompt tokens summed over the
  requests of one trial were about 19,000 on average and 70,700 at most in Exp 2 (21,000 and
  72,400 in Exp 1). Give the model a context window of at least 32,768 tokens; a smaller
  window makes long trials fail or, on servers that truncate silently, makes the model lose
  the task description.
- **Output.** Each turn may produce up to 4,096 tokens (8,192 at `--effort high` and above).
- **Answer format.** The final message must contain a JSON object, for example
  `{"support": [3, 17, 29], "signs": [1, -1, 1]}` in Exp 2. The parser takes the last
  object of the right shape anywhere in the text, so reasoning before it is harmless.

## 2. Expected requests, tokens and time

Taken from the paper's records of the two preset conditions (13 models in Exp 2, 5 in Exp 1):

| Preset | Trials per model | Requests per trial (mean / max) | Prompt tokens per trial (mean / 90th pct.) | Output tokens per trial (mean / 90th pct.) |
| --- | ---: | ---: | ---: | ---: |
| `smoke` | 3 | 4.4 / 5 | 19,200 / 35,700 | 1,950 / 4,250 |
| `paper-exp2-main` | 150 | 4.4 / 5 | 19,200 / 35,700 | 1,950 / 4,250 |
| `paper-exp1` | 192 | 5.0 / 6 | 21,100 / 39,400 | 3,250 / 6,700 |

Output tokens include reasoning tokens. Multiply by the number of trials for a total: the
main Exp 2 condition is roughly 2.9 million prompt tokens and 0.3 million output tokens
per model. For a hosted API that is the cost estimate; for a local model, divide by your
server's prompt-processing and generation speeds (a server that reuses the cached prefix
of the conversation processes far fewer prompt tokens). At 20 output tokens per second the
three `smoke` trials take a few minutes and `paper-exp2-main` takes several hours.
Agent-written Python runs in the sandbox with a 10 s (Exp 2) or 15 s (Exp 1) limit per call.

## 3. Start a server and run the smoke test

The `smoke` preset runs three trials of the main Exp 2 condition (seed 5000: black box,
clean code, hard-guard code). It checks that the server, the tool calls and the answer
format work before you spend hours on the full condition. Replace `<model>` with the model
name your server expects; it is sent as `model` and becomes part of every run key.

**Ollama.** Start the server with a large enough context (the default is small and long
prompts are cut silently):

```bash
OLLAMA_CONTEXT_LENGTH=32768 ollama serve          # in another terminal; then: ollama pull <model>
python -m blackwell_inversion --preset smoke --provider openai \
    --base-url http://127.0.0.1:11434/v1 --models <model> --out results/my_model.jsonl
```

**vLLM.** Tool calls need the automatic tool choice and the parser for your model family:

```bash
vllm serve <model> --enable-auto-tool-choice --tool-call-parser <parser> --max-model-len 32768
python -m blackwell_inversion --preset smoke --provider openai \
    --base-url http://127.0.0.1:8000/v1 --models <model> --out results/my_model.jsonl
```

**llama.cpp server.** `--jinja` enables the chat template that carries tool calls:

```bash
llama-server -m model.gguf --jinja -c 32768 --port 8080
python -m blackwell_inversion --preset smoke --provider openai \
    --base-url http://127.0.0.1:8080/v1 --models <model> --out results/my_model.jsonl
```

**LM Studio.** Load the model with a context length of at least 32,768, start the local
server (default port 1234), then:

```bash
python -m blackwell_inversion --preset smoke --provider openai \
    --base-url http://127.0.0.1:1234/v1 --models <model> --out results/my_model.jsonl
```

**A hosted OpenAI-compatible API.** Name the environment variable that holds the key, give
the prices so that the spend cap works, and set a cap:

```bash
export MY_PROVIDER_KEY=...        # never put the key itself on the command line
python -m blackwell_inversion --preset smoke --provider openai \
    --base-url https://api.example.com/v1 --api-key-env MY_PROVIDER_KEY \
    --input-price-per-mtok 0.50 --output-price-per-mtok 1.50 --max-cost-usd 2 \
    --models <model> --out results/my_model.jsonl
```

The key is read from that variable (or from a line `MY_PROVIDER_KEY=...` in `~/.env`),
sent as a bearer token, and never written to the log or to an error message. A variable
that is not set stops the run at start-up. Without `--api-key-env` no key is sent, which
is what local servers expect. A key sent over plain `http://` to a host other than this
machine prints a warning: use `https://`. `--models` is required with `--provider openai`
(it is the name sent to your server). Requests to a loopback address (`127.0.0.1`,
`localhost`, `::1`) never use the proxy set in `http_proxy`; other hosts do.

**Options of `--provider openai`:**

| Option | Default | Meaning |
| --- | --- | --- |
| `--base-url` | (required) | Server URL up to `/v1`; requests go to `{base-url}/chat/completions`. |
| `--api-key-env VAR` | none | Environment variable holding the key. |
| `--input-price-per-mtok`, `--output-price-per-mtok` | 0 | USD per million tokens, for `cost_usd` and `--max-cost-usd`. 0 suits a local model. |
| `--max-tokens-field` | `max_tokens` | Use `max_completion_tokens` for servers that reject `max_tokens`. |
| `--effort` | `medium` | Sent as `reasoning_effort`; `--effort ""` sends nothing. |
| `--request-timeout-s` | 600 | Seconds to wait for one reply; slow local models may need more. A reply that does not arrive in time is not retried (see §8). |
| `--workers` | 1 | Concurrent trials; raise it only if your server serves requests in parallel. |
| `--max-consecutive-errors` | 5 | Stop after this many trial errors in a row; 0 never stops. |
| `--debug` | off | Print the traceback of every trial error. |

If the server rejects `reasoning_effort` (an HTTP 400 that names it), add `--effort ""`. A
preset accepts this override and prints a note: the run keys then read `ef=default`. The
summary shows such a run as its own condition (effort=default), checks its preset
coverage against those run keys, and compares it with the paper's rows at effort=default
(Sonnet 4.5 and Haiku 4.5, which ran without an effort setting). Thinking mode and
thinking budget are not sent to OpenAI-compatible servers. The module docstring of
`blackwell_inversion/clients/openai_compat.py` lists how the client reads the reply
variants that real servers produce (missing usage, `content: null`, arguments that are not
JSON, missing tool-call ids, reasoning fields) and which fields of the server's reply it
sends back on the next request (all of them, including `extra_content` on tool calls and
`reasoning_content`, `reasoning` or `reasoning_details`, which some servers require).

## 4. Run the paper's conditions

```bash
python -m blackwell_inversion --preset paper-exp2-main --provider openai \
    --base-url http://127.0.0.1:11434/v1 --models <model> --out results/my_model.jsonl
python -m blackwell_inversion --preset paper-exp1 --provider openai \
    --base-url http://127.0.0.1:11434/v1 --models <model> --out results/my_model.jsonl
```

`python -m blackwell_inversion --list-presets` prints every option a preset sets.

| Preset | Condition |
| --- | --- |
| `paper-exp2-main` | Exp 2, Majority-3, standard protocol, medium effort, 5 turns per mode, seeds 5000-5024; black box and the five white-box variants (clean code, short control, refutable control, hard-guard code, opaque weights). 150 trials. |
| `paper-exp1` | Exp 1, hardened 4-XOR family (`4xor_strict`), Python tool, hard and solvable regimes; black box, clean code, sparse weights and dense weights; 12 planted/clean pairs. 192 trials. |
| `smoke` | Three trials of `paper-exp2-main`. |

A preset fixes every option that defines the condition; giving one of them with another
value is an error, so a preset run is always the paper's condition. The values were read
from `data/main/` and `tests/test_presets.py` checks them against the logged records.
The `smoke` run keys are a subset of `paper-exp2-main`, so the full run with the same
`--out` file continues after them. Every run is resumable: rerunning the same command
skips the trials already in the log, so you can stop with Ctrl-C at any time. Queued
trials are cancelled, requests in flight finish or time out, a trial cut short is not
written, and every trial already written stays in the log.

**Exit status.** 0 when every scheduled trial is in the log; 1 when a trial failed or the
run was stopped after `--max-consecutive-errors` errors in a row; 2 for a usage error; 3
when the spend cap (or an HTTP 402 from the server) stopped the run; 130 after Ctrl-C.
Rerun the same command after 1, 3 or 130 to complete the log.

## 5. Read the summary

```bash
python -m blackwell_inversion.analysis.summarize results/my_model.jsonl \
    --preset paper-exp2-main --compare-paper
```

For each model and condition the summary shows:

- **Exp 2:** black-box accuracy, and per white-box variant the held-out accuracy (`acc`),
  `Delta_WB = WB - BB` with its paired bootstrap 95% CI and one-sided sign-flip p-value
  (greater for clean code, less otherwise), the **trap rate** (share of answers equal to
  the decoy rule), the parse-failure rate, and the mean turns, oracle queries, tokens per
  trial and the cost. A negative `Delta_WB` on hard-guard code with a high trap rate is the
  computational inverse Blackwell effect: the code made the model worse than the oracle
  alone.
- **Exp 1:** per regime and access mode the distinguishing advantage `Adv = TPR - FPR` with
  its Newcombe-Wilson 95% CI and the exact Fisher p-value. In the hard regime no auditor
  should exceed chance; in the solvable regime reading the code should help.

The statistics are computed by the same functions as the paper's report
(`analysis/main_report.py`, seeds included), so for the paper's own logs the summary
reproduces the report's numbers (checked by `tests/test_summarize.py`).

**Incomplete logs.** An interrupted run leaves conditions in which some modes or variants
lack seeds that others have. The summary names the missing seeds and withholds `Delta_WB`
for such a condition, because the estimate would be on a different set of seeds than the
black box; `--allow-incomplete` prints it over the seeds that do pair, marked with `*N`.
With a single seed (the `smoke` preset) only the point value of `Delta_WB` is shown: one
pair gives no interval and no p-value. `--preset NAME` also lists the scheduled trials
that are not in the log, for the thinking mode and effort found in the log (so a run with
`--effort ""` is checked against its own run keys). Unreadable lines (an interrupted
write) are counted and skipped; a repeated trial (the same run key, or the same
condition, mode, variant and seed under another run key) keeps its first record.
`--out-json FILE` writes the whole summary as JSON.

**Conditions are never pooled.** A condition is everything the run key encodes apart from
the access mode, the variant and the seed: model, effort, thinking mode, protocol, budget
track, task type and token cap. Runs of one model with different settings in one log
(for example `smoke`, `--thinking-mode disabled` and `--protocol verify_first`) are
summarised as separate cells, and each cell header names its thinking mode and protocol.

## 6. Compare with the paper

`--compare-paper` prints, under each of your conditions, the paper's models in the same
condition from `reports/report_c3_unified.json`: their black-box accuracy, `Delta_WB` per
variant and the trap rate on hard-guard code (for Exp 1, their `Adv` per cell). 13 of the
paper's 15 models ran the main Exp 2 condition (`paper-exp2-main`), and 5 have Exp 1 cells
in the hardened 4-XOR family that `paper-exp1` uses. The rows have the same effort as your log; for an effort the paper
did not run, the medium-effort rows are shown with a note, and a note also marks a
thinking mode that differs from the paper's. The main report has the standard protocol
only, so a cell of another protocol gets no rows. The report is read from the checkout; if
it is not found (for example after a non-editable `pip install .`), the command stops
and names `--paper-report`, which points at the file. Keep in mind:

- The paper ran 25 seeds per model in the main Exp 2 condition (Gemma 4 31B 15, Gemma 4
  26B 10, Gemini 3.1 Flash-Lite 20). With 25 seeds the paper's 95% CIs of `Delta_WB` are
  about ±0.05 to ±0.15 wide on each side, so differences smaller than that are not resolved.
- In Exp 1 some paper cells have more than 12 pairs; their CIs are narrower than yours.
- The paper's models ran through their providers' native APIs with their own reasoning
  settings; your model runs through an OpenAI-compatible server with whatever chat
  template and tool-call parser the server uses. Differences in tool-call handling are
  part of what you measure.

## 7. A client of your own

Anything that can map a conversation to a reply can be evaluated: an in-process model, a
proprietary API, a rule-based baseline. Write a `ModelClient` subclass (the contract is in
[`blackwell_inversion/clients/base.py`](../blackwell_inversion/clients/base.py)) and a
factory that takes the parsed command line:

```bash
python -m blackwell_inversion --client examples.rule_based_client:make_client \
    --preset smoke --models toy --out results/toy.jsonl
python -m blackwell_inversion --client path/to/my_client.py:make_client --preset smoke --models mine --out results/mine.jsonl
```

A module path (`package.module:factory`) is resolved from the working directory, as with
`python -m`, also when the installed `blackwell-inversion` command is used.

[`examples/rule_based_client.py`](../examples/rule_based_client.py) is a complete, runnable
example without dependencies. Your client receives the conversation in Anthropic Messages
form and returns `content` blocks, a `stop_reason`, `usage` and the call's cost; call
`check_budget()` before and `record_usage(...)` after each call so that `--max-cost-usd`
works. The factory gets the full `argparse` namespace (`args.models`, `args.max_cost_usd`,
...). An import error, an error while the module is loaded (a `SyntaxError`, a
`NameError`, ...), a missing factory, a factory that raises or one that does not return a
`ModelClient` stops the run before any trial with a one-line message saying which.

## 8. Limitations and technical notes

- **The sandbox is not a security boundary.** The harness runs Python written by the model
  under test on your machine. It runs in a child process with a temporary working
  directory, an allowlisted environment (no API keys), CPU, memory and time limits, but as
  your user: it can read any file you can read (including `~/.env`) and it has network
  access. Run evaluations of untrusted models in a disposable VM or container.
- **The spend cap is an estimate per invocation.** It prices the token counts the server
  reports with the prices you give, and it starts at zero in every invocation; spend of
  earlier, resumed runs is not counted. A server that reports no usage is counted as 0
  tokens, so the cap cannot stop it. With `--workers N` requests in flight may exceed the
  cap by up to N requests.
- **Trial errors.** A trial that fails (a reply the client cannot interpret, an HTTP error
  such as a context-length 400, a 5xx that persists after the retries, an unreachable
  server) is reported as one line `[TRIAL ERROR] <run_key>: <message>` (`--debug` adds the
  traceback). No record is written for it, the run continues with the next trial, and the
  next run retries it. After `--max-consecutive-errors` errors in a row (default 5) the run
  stops, since the setup is then likely broken (a server that is down, a context window
  that is too small). The exit status is 1 in both cases.
- **Timeouts and retries.** 429 and 5xx replies and connection errors (refused, reset, a
  connect timeout) are retried up to five attempts with backoff. A reply that does not
  arrive within `--request-timeout-s` is not retried: the server may still be generating,
  and a retry would repeat the generation (on a hosted API, a billed one the spend cap does
  not see). The error asks for a larger `--request-timeout-s`.
- **Ctrl-C.** No new trial starts, queued trials are cancelled, a request in flight
  finishes or times out, and the trials already written stay in the log (exit status 130).
- **Malformed tool calls** (arguments that are not a JSON object, a non-integer `count` or
  `seed`, a `seed` too negative for the random-query generator, code that is not valid
  UTF-8 text) are answered with an error message to the model and use no query budget.
- **Tool calls on the final turn.** The last turn is sent with `tool_choice: "none"`. A
  server that ignores it may still return tool calls; they are executed and the trial
  ends, so the answer must already be in that turn's text.
- **Run keys.** The run key does not encode `--thinking-budget`; runs that differ only in it
  share keys and resume into each other. With `--max-tokens-override`, the
  `max_tokens_override` field is also written to Exp 1 records, although only Exp 2 uses
  the override. Both are kept as they are because the published records depend on them.
- **Model drift.** A hosted model behind a fixed name can change; record the date and the
  server version with your results.
