# Evaluating your own model

This guide runs the benchmark on a model of your choice and puts its result next to the
paper's. It covers a local model behind an OpenAI-compatible server, a hosted
OpenAI-compatible API, and a client of your own. Everything below runs from the
repository root after the setup in the [README](../README.md#quick-start).

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
sent as a bearer token, and never written to the log or to an error message. Without
`--api-key-env` no key is sent, which is what local servers expect.

**Options of `--provider openai`:**

| Option | Default | Meaning |
| --- | --- | --- |
| `--base-url` | (required) | Server URL up to `/v1`; requests go to `{base-url}/chat/completions`. |
| `--api-key-env VAR` | none | Environment variable holding the key. |
| `--input-price-per-mtok`, `--output-price-per-mtok` | 0 | USD per million tokens, for `cost_usd` and `--max-cost-usd`. 0 suits a local model. |
| `--max-tokens-field` | `max_tokens` | Use `max_completion_tokens` for servers that reject `max_tokens`. |
| `--effort` | `medium` | Sent as `reasoning_effort`; `--effort ""` sends nothing. |
| `--request-timeout-s` | 600 | Seconds to wait for one reply; slow local models may need more. |
| `--workers` | 1 | Concurrent trials; raise it only if your server serves requests in parallel. |

If the server rejects `reasoning_effort` (an HTTP 400 that names it), add `--effort ""`. A
preset accepts this override and prints a note: the run keys then read `ef=default`, and
the summary shows the paper's medium-effort rows for comparison. Thinking mode and
thinking budget are not sent to OpenAI-compatible servers. The module docstring of
`blackwell_inversion/clients/openai_compat.py` lists how the client reads the reply
variants that real servers produce (missing usage, `content: null`, arguments that are not
JSON, missing tool-call ids, reasoning fields).

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
skips the trials already in the log, so you can stop with Ctrl-C at any time (a trial cut
short is not written).

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
`--preset NAME` also lists the scheduled trials that are not in the log. Unreadable lines
(an interrupted write) and repeated run keys are counted and skipped. `--out-json FILE`
writes the whole summary as JSON.

## 6. Compare with the paper

`--compare-paper` prints, under each of your conditions, the paper's models in the same
condition from `reports/report_c3_unified.json`: their black-box accuracy, `Delta_WB` per
variant and the trap rate on hard-guard code (for Exp 1, their `Adv` per cell). Keep in
mind:

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

[`examples/rule_based_client.py`](../examples/rule_based_client.py) is a complete, runnable
example without dependencies. Your client receives the conversation in Anthropic Messages
form and returns `content` blocks, a `stop_reason`, `usage` and the call's cost; call
`check_budget()` before and `record_usage(...)` after each call so that `--max-cost-usd`
works. The factory gets the full `argparse` namespace (`args.models`, `args.max_cost_usd`,
...). An import error, a missing factory or a factory that does not return a `ModelClient`
stops the run before any trial with a message saying which.

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
- **A reply the client cannot interpret** (not JSON, an error object, no choices) stops a
  serial run with an error; no record is written for that trial, and rerunning resumes.
  With `--workers` the trial is counted as an error and the run continues.
- **Malformed tool calls** (arguments that are not a JSON object, a non-integer `count` or
  `seed`) are answered with an error message to the model and use no query budget.
- **Tool calls on the final turn.** The last turn is sent with `tool_choice: "none"`. A
  server that ignores it may still return tool calls; they are executed and the trial
  ends, so the answer must already be in that turn's text.
- **Run keys.** The run key does not encode `--thinking-budget`; runs that differ only in it
  share keys and resume into each other. With `--max-tokens-override`, the
  `max_tokens_override` field is also written to Exp 1 records, although only Exp 2 uses
  the override. Both are kept as they are because the published records depend on them.
- **Model drift.** A hosted model behind a fixed name can change; record the date and the
  server version with your results.
