# Architecture

## Modules

| Module | Role |
| --- | --- |
| `generators.py` | Builds instances. `generate_kxor_instance` (Exp 1) returns a `KXorInstance` that exports clean code, obfuscated code or ReLU weights; `generate_blackwell_inv_instance` (Exp 2) returns a `BlackwellInvInstance` with the hidden rule, the decoy and the white-box artifact of each variant, plus 100 held-out test points. Everything is a deterministic function of the seed. |
| `validators.py` | Independent re-evaluation of exported code and weights. |
| `baselines.py` | Classical attacks, information-theoretic bounds, and `score_rule_prediction`, the Exp 2 scorer. |
| `sandbox.py` | `MeteredOracle` answers queries in the parent process and enforces the budget; `IsolatedPythonSandbox` runs agent code in a child process inside a temporary workspace. |
| `clients/` | `base.ModelClient` (the contract below), `pricing.py`, and the Anthropic, Gemini, OpenAI-compatible (`openai_compat.py`) and mock implementations. |
| `agent.py` | Tool schemas and `run_agent_loop`, the provider-independent tool loop. |
| `exp1.py`, `exp2.py` | One trial each: build the instance and the prompt, run the loop, parse and score the answer, return the record. `exp2.py` also holds the budget tracks and the protocols. |
| `parsing.py` | Final-answer parsers. |
| `schedule.py` | Job lists, run keys and the resumable JSONL log. |
| `presets.py` | Named sets of command line options for the paper's conditions (`--preset`). |
| `cli.py` | Argument parsing, presets, client choice (`--provider`, `--client`), serial or threaded execution. |
| `analysis/` | Reports and figures computed from the logs only; `summarize.py` summarises any log with the report's estimators. |

## Data flow

```
schedule (run_key, condition, seed)
   |
   v
exp1 / exp2 trial ---- generators: instance from (condition, seed)
   |                          |
   |                          +--> sandbox workspace: artifact.py or weights.npz (WB only), queries.json
   |                          +--> MeteredOracle (parent process): ground truth, query budget
   v
agent loop  <-->  ModelClient.create_message  <-->  provider API (or mock)
   |   tool calls: query_oracle_random / query_oracle -> MeteredOracle
   |               run_python                         -> IsolatedPythonSandbox
   v
final text -> parsing -> scoring (baselines.score_rule_prediction for Exp 2)
   |
   v
record (one JSON line, appended under its run_key) -> data/*.jsonl
   |
   v
analysis: main_report, mechanism, followup -> reports/*.json, *.txt -> figures
```

The ground truth never leaves the parent process: the model sees only the prompt, the
oracle's answers and whatever its own code prints. The instance, its held-out points and
the harness-check points are seeded from the trial seed, and `query_oracle_random` from the
seed in the tool call and the queries already used, so a dry run is deterministic apart
from timing fields.

## The record

`exp1.run_exp1_trial` and `exp2.run_exp2_trial` return flat dictionaries; the driver adds
`run_key` (and `max_tokens_override` when set) and appends the record as one JSON line.
The fields are listed in [`../data/README.md`](../data/README.md); the run key format is in
`schedule.py`. Field names and label values are part of the published data and must not
change.

## The client contract

`blackwell_inversion/clients/base.py` documents it in full. In short, a client is a
`ModelClient` subclass that implements

```python
def create_message(self, *, model, system, messages, max_tokens, tools=None,
                   tool_choice=None, thinking_budget=0, thinking_mode="adaptive",
                   effort="") -> dict
```

- **Input** is in Anthropic Messages form: alternating user/assistant `messages` whose
  content is a string or a list of `text`, `tool_use` and `tool_result` blocks; `tools` with
  JSON-Schema `input_schema`; `tool_choice={"type": "none"}` on turns where tools are not
  allowed. The tool list is the same on every turn.
- **Output** is `{"content": [...], "stop_reason": ..., "usage": {...}, "_cost_usd": float}`.
  The loop reads `text` and `tool_use` blocks, treats `stop_reason == "tool_use"` as a tool
  turn and `"max_tokens"` as a truncated turn, and stores `content` verbatim as the
  assistant turn (the Gemini client uses this to replay its raw parts).
- **Accounting:** call `check_budget()` before sending (it raises `BudgetExceededError` once
  the cap is reached) and `record_usage(...)` after a response; price tokens with
  `pricing.estimate_cost_usd`.
- **Selection:** `cli.make_client` returns the client of `--client module:factory` if given
  (the factory is called with the parsed arguments and must return a `ModelClient`), else
  the one named by `--provider`; `auto` keeps the original rule (mock for `--dry-run`, Gemini
  if every model id is a Gemini or Gemma id, else Anthropic).

The Gemini and OpenAI-compatible clients are examples of translating this contract to a
different wire format: they convert messages and tool schemas on the way out and
normalise the reply on the way back, so nothing outside `clients/` knows which provider is
in use. `openai_compat.py` documents in its module docstring how each irregular reply of a
real server is read.

## Notes for users of the code

- **Malformed tool calls.** A tool call whose input is not an object, or whose `count` or
  `seed` cannot be read as an integer, gets an error `tool_result` and uses no query
  budget. Before, such a call raised `ValueError` and ended the run, so no logged trial
  contains one; the prompts and tool schemas are unchanged.
- **Interrupted log writes.** `append_jsonl` starts a record on a new line if the log does
  not end in one, so a record written after a partial line is not lost. Readers skip the
  partial line.
- **Run keys.** The run key does not encode `--thinking-budget`, so runs that differ only in
  it share keys. With `--max-tokens-override`, the `max_tokens_override` field is also
  added to Exp 1 records, although only Exp 2 uses the override. Both are left as they are
  because the published records were written this way.
- **Spend cap.** `--max-cost-usd` caps an estimate for one invocation; see the README.
- **Mock client.** The dry-run client recognises Exp 2 by the answer-format line that every
  Exp 2 task type shares. Its strategies are written for Majority-3, so `dlist3` and
  `dtree2` dry runs now parse but are seldom right; Majority-3 and Exp 1 dry runs are
  unchanged.
- **Sandbox.** Agent code runs as your user; it is not a security boundary (README).
