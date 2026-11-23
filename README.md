# The Computational Inverse Blackwell Effect: Experiment Code and Trial Logs

This repository holds the code, the raw logs of 7,631 live trials, and the analysis
scripts behind the paper "The Computational Inverse Blackwell Effect and the
Observer-Relative Virtual Black-Box Property in Tool-Using Large Language Model Auditors"
(under review). It also lets you run the same benchmark on a model of your own.

## What the benchmark measures

An LLM agent acts as an auditor of a small Boolean function. It is tested in two access
modes, with the same seeds and the same interaction budget:

- **Black box (BB):** the agent can only query the function (send inputs, read outputs).
- **White box (WB):** the agent has the same query interface plus the function's source
  code or neural-network weights.

The difference in success, `Delta_WB = WB - BB`, is the white-box advantage. Intuition says
it should never be negative: extra information cannot hurt. A negative value is the
computational inverse Blackwell effect: the agent does worse with the code than with the
oracle alone. Two compilers produce functions on which the advantage is close to zero or
negative, without changing what the function computes:

- **Experiment 1 (`exp1`, compiler `C_kXOR`):** decide whether the function hides a planted
  backdoor (a noisy k-XOR trigger) or is clean. Metric: distinguishing advantage
  `Adv = TPR - FPR`. In the hard regime the backdoor is built to stay hidden even from an
  agent that reads the code; in the solvable regime the code gives it away.
- **Experiment 2 (`exp2`, compiler `C_inv`):** recover a hidden 3-variable rule. The white-box
  code contains a readable decoy branch that never fires. Metrics: held-out accuracy,
  `Delta_WB`, and the trap rate (how often the answer equals the decoy rule).

## Evaluate your own model

Any model behind an OpenAI-compatible server (a local model served by Ollama, vLLM, the
llama.cpp server or LM Studio, or a hosted API) can be run on the paper's conditions and
compared with the paper's 15 models:

```bash
python -m blackwell_inversion --preset smoke --provider openai \
    --base-url http://127.0.0.1:11434/v1 --models <model> --out results/my_model.jsonl     # 3 trials
python -m blackwell_inversion --preset paper-exp2-main --provider openai \
    --base-url http://127.0.0.1:11434/v1 --models <model> --out results/my_model.jsonl     # 150 trials
python -m blackwell_inversion.analysis.summarize results/my_model.jsonl --compare-paper
```

`--client module:factory` plugs in any other client (an in-process model, a proprietary
API); [`examples/rule_based_client.py`](examples/rule_based_client.py) is a runnable
example. [`docs/evaluating_your_model.md`](docs/evaluating_your_model.md) walks through
server setup, what the model must support, the expected number of requests and tokens,
how to read the summary and how to compare with the paper.

## Quick start

Linux is required (the sandbox uses the `resource` module). The results were produced with
Python 3.14.4, numpy 2.3.5, scipy 1.16.3 and matplotlib 3.10.7; any Python 3.11 or later
with these pinned versions should work. The model clients use only the standard library.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # pinned versions; or: pip install -e ".[figures]"

python -W error -m unittest discover -s tests -t .    # 156 tests, about 1.5 minutes, no network

# Run the whole pipeline with the zero-cost mock client (about 1 second)
python -m blackwell_inversion --dry-run --experiment exp2 --models claude-haiku-5-5 \
    --exp2-seeds 1 --out /tmp/dry.jsonl
```

`pip install -e .` also installs a `blackwell-inversion` command equivalent to
`python -m blackwell_inversion`. The old entry point `python -m blackwell_inversion.runner`
still works.

## Repository map

| Path | Contents |
| --- | --- |
| `blackwell_inversion/generators.py` | The two compilers: `C_kXOR` instances (clean code, MBA-obfuscated code, 2- and 3-layer ReLU weights) and `C_inv` instances (five white-box variants). Alarm thresholds and false-alarm bounds are computed here. |
| `blackwell_inversion/validators.py` | Checks, independent of the generators, that exported code and weights compute the same function in every variant. |
| `blackwell_inversion/baselines.py` | The eight classical reference attacks (weight statistics, SVD, Canon+Kikuchi spectral solver, simulated annealing, ...), black-box advantage bounds and the Exp 2 scorer. |
| `blackwell_inversion/sandbox.py` | `IsolatedPythonSandbox` (runs agent-written Python) and `MeteredOracle` (enforces the query budget). |
| `blackwell_inversion/clients/` | The `ModelClient` contract (`base.py`), the price table, and the Anthropic, Gemini, OpenAI-compatible and mock clients. |
| `blackwell_inversion/agent.py` | Tool schemas and the tool-calling agent loop. |
| `blackwell_inversion/exp1.py`, `exp2.py` | Trial functions, prompts, regimes, budget tracks and the Exp 2 protocols. |
| `blackwell_inversion/parsing.py` | Parsers for the agent's final answer. |
| `blackwell_inversion/schedule.py`, `cli.py` | Trial schedules and run keys; the command line driver with the resumable JSONL log. |
| `blackwell_inversion/presets.py` | The `--preset` conditions (`paper-exp2-main`, `paper-exp1`, `smoke`). |
| `blackwell_inversion/analysis/` | `stats.py` (statistics), `main_report.py`, `mechanism.py`, `followup.py` (the three reports), `summarize.py` (summary of your own log), `figures.py`, and `legacy_report.py` (the earlier stage report). |
| `data/` | Trial logs, `SHA256SUMS`, and `data/README.md` with the field dictionary. |
| `reports/` | The three reports the paper relies on; `reports/legacy/` holds superseded stage reports. |
| `docs/` | [`evaluating_your_model.md`](docs/evaluating_your_model.md), [`protocol.md`](docs/protocol.md) (study protocol), [`architecture.md`](docs/architecture.md), [`reproducing.md`](docs/reproducing.md). |
| `examples/` | A runnable custom client for `--client`. |
| `tests/` | Unit, integration and reproduction tests (`unittest`). |

## Data

The paper's 7,631 trials are in three files; one JSON record per line. See
[`data/README.md`](data/README.md) for every file, the merge of the C3 sources and the field
dictionary.

| File | Trials | Contents | SHA-256 |
| --- | ---: | --- | --- |
| `data/calibration/results_c1_haiku55.jsonl` | 680 | Parametric difficulty calibration (Haiku 5.5) | `045dc5c6c30d7e88e3de6efcaa57be466b7bf8c00a9b06fe9062076509fb1b00` |
| `data/main/results_c2_pilot.jsonl` | 5,026 | Nine Anthropic models: 2,064 Exp 1 and 2,962 Exp 2 trials | `ba07b3d6c6a31bf478b04130fc062c06b0d74217511e8e923f69bb4a9146c6fc` |
| `data/main/results_c3_expansion.jsonl` | 1,925 | Google Gemini, Gemma 4 and causal controls: 508 Exp 1 and 1,417 Exp 2 trials | `5800f0c81315ff5b6e88303d834997899a815882eab26ec22009ec8134e498c6` |

Follow-up runs made after the main data set was locked (not part of the 7,631):

| File | Trials | Contents | SHA-256 |
| --- | ---: | --- | --- |
| `data/followup/results_c4_refutable.jsonl` | 100 | The refutable control on four models that lacked it | `1fa4d54173f8e033ebf9985e5dd761407f0ac4e02f4f490351214997a1a8b280` |
| `data/followup/results_c4_protocols.jsonl` | 1,200 | Three protocols x four conditions x four models x 25 seeds | `150e70128d7c9e93174ed165f3a7d5d54cf7757ff6a652c6a10dde08cd72c5bf` |
| `data/followup/results_c5_check_tools.jsonl` | 163 | The answer check with tools; one model incomplete (see `data/README.md`) | `b0d80dab17b0638324327e15ca42adbce2e2a41c6dbd1aee97c780029eb59517` |

Verify all logs (including the ten C3 source files) with:

```bash
cd data && sha256sum -c SHA256SUMS
```

## Reproducing the numbers

None of these steps calls an API. Exact commands and measured runtimes are in
[`docs/reproducing.md`](docs/reproducing.md).

```bash
python -m blackwell_inversion.analysis.main_report --out-json /tmp/report.json > /tmp/report.txt
cmp /tmp/report.json reports/report_c3_unified.json     # no output means byte-identical
python -m blackwell_inversion.analysis.mechanism --out-json /tmp/mechanism.json > /tmp/mechanism.txt
cmp /tmp/mechanism.json reports/report_mechanism.json
python -m blackwell_inversion.analysis.followup --out-json /tmp/followup.json > /tmp/followup.txt
cmp /tmp/followup.json reports/report_followup.json

RUN_SLOW=1 python -m unittest tests.test_reproducibility   # all of the above, plus figures

python -m blackwell_inversion.analysis.figures --lang en    # figures/ (needs matplotlib)
```

## Running live experiments

Live runs are paid. Keys are read from the environment (or from `~/.env`) and are never
stored in the repository or passed to the sandbox:

```bash
export ANTHROPIC_API_KEY=...   # Claude models
export GEMINI_API_KEY=...      # Gemini and Gemma models
python -m blackwell_inversion --experiment exp2 --models claude-haiku-5-5 \
    --effort medium --exp2-seeds 25 --max-cost-usd 5 --out results_new.jsonl
```

With the default `--provider auto`, the Gemini client is used if every model id starts with
`gemini-` or `gemma-`, otherwise the Anthropic client; one run uses one provider.
`--provider openai` selects the OpenAI-compatible client and `--client` a client of your
own (see [`docs/evaluating_your_model.md`](docs/evaluating_your_model.md)).
`python -m blackwell_inversion --help` lists all options; `--list-models` lists the
Anthropic models visible to your key.

**The spend cap.** `--max-cost-usd` caps the estimated spend of one invocation. Each
response's token usage is priced with the table in `blackwell_inversion/clients/pricing.py`
(unknown model ids get a deliberately high fallback rate; with `--provider openai` the
prices given by `--input-price-per-mtok` and `--output-price-per-mtok` are used, default 0),
and before every request the
client compares the running total with the cap; once the total has reached it, the client
raises `BudgetExceededError` instead of sending the request. A serial run then stops; with
`--workers N` no new trial starts, and requests already in flight may still complete, so
the total can exceed the cap by up to N requests. A trial interrupted by the cap is not
written to the log. The total is an estimate, not the provider's bill, and it starts at
zero in every invocation: spend from earlier, resumed runs is not counted. An HTTP 402
(prepaid credit exhausted) from the Gemini or an OpenAI-compatible API is treated like the cap.

**Resuming.** Every record carries a `run_key` that encodes the model, the settings and the
condition. Restarting with the same `--out` file skips every trial already logged;
`--resume-from a.jsonl,b.jsonl` also skips keys found in other logs.

**Exp 2 protocols.** `--protocol verify_first` tells the agent to test its candidate on the
oracle; `--protocol harness_check` checks the answer on ten fresh queries and grants one
tool-free revision turn on a mismatch; `--protocol harness_check_tools` grants three
revision turns, two with tools. `--log-tool-inputs` adds the agent's Python code and its
queried inputs to each record. Runs with these protocols are not part of the 7,631 trials.

## Evaluated models

- **Anthropic (9):** `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251001`, `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-opus-4-8`, `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`, `claude-fable-5-2`
- **Google Gemini (4):** `gemini-3.1-flash-lite`, `gemini-3.5-flash-lite`, `gemini-3.8-flash`, `gemini-4-carbon`
- **Open-weight Gemma 4 (2):** `gemma-4-26b-a4b-it`, `gemma-4-31b-it`

## Names in the paper and labels in the data

The paper names conditions in words; the code and the logs use the labels below.

| Name in the paper | Record field | Label |
| --- | --- | --- |
| Hard regime | `regime` | `vbb_hard` |
| Solvable regime | `regime` | `transparent` |
| Keyless control (leak test, η = 0.50) | `regime` | `control_impossible` |
| 4-XOR, n = 40 and n = 80 (calibration families) | `family`, `n` | `4xor` (`n` = 40 or 80) |
| 3-XOR family | `family` | `3xor` |
| Hardened 4-XOR family | `family` | `4xor_strict` |
| Fixed-size 4-XOR family (n = 120) | `family` | `4xor_fixed120` |
| Clean code (Exp 1) | `substrate` | `code_clean` |
| Obfuscated code | `substrate` | `code_obf` |
| Sparse weights (two layers) | `substrate` | `relu_weights` |
| Dense weights (three layers, orthogonal input layer) | `substrate` | `relu_weights_dense` |
| With / without the Python tool | `track` | `tool_python` / `tool_free` |
| Majority-3 task | `task_type` | `majority3` |
| Decision List-3 task | `task_type` | `dlist3` |
| Clean code (Exp 2) | `variant` | `wb_clean` |
| Short control (short dead-code guard) | `variant` | `wb_decoy_trivial_guard` |
| Refutable control (same byte length as the hard-guard code) | `variant` | `wb_decoy_long_refutable_guard` |
| Hard-guard code | `variant` | `wb_inv_code` |
| Opaque weights | `variant` | `wb_inv_relu` |
| 3/3-turn tight budget | `budget_track` | `matched_total_budget` |
| 5/5-turn medium budget | `budget_track` | `matched_medium_budget` |
| 4/8-turn asymmetric budget | `budget_track` | `matched_query_free_code` |
| 8/8-turn wide equal budget | `budget_track` | `matched_wide_equal_budget` |
| Output token cap of 8,192 | `max_tokens_override` | `8192` |
| Verify-first instruction protocol (follow-up) | `protocol` | `verify_first` |
| Answer-check protocol (follow-up) | `protocol` | `harness_check` |
| Answer check with tools (follow-up) | `protocol` | `harness_check_tools` |

The code names of the eight classical attacks (`scalar_b2`, `deg1`, `deg2_svd`,
`deg2_bias_svd`, `probe_m3`, `canon_short_cover`, `canon_kikuchi`, `canon_sa`) are in
`baselines.py`; the agent's tools (`query_oracle_random`, `query_oracle`, `run_python`) are
defined in `agent.py`.

## Security notes on the sandbox

The harness executes Python code written by the model under test. `IsolatedPythonSandbox`
runs it in a child process with a fresh temporary working directory, an allowlisted
environment (no API keys or other host variables), CPU-time and address-space limits and a
wall-clock timeout. The ground truth stays in the parent process. This keeps the agent from
reading the answer or the keys through the environment, but it is not a security
boundary: the child runs as your user, can read any file your user can read (including
`~/.env`) and has network access. Run live experiments in a disposable VM or container if
that matters to you. The resource limits need Linux.

## License

To be announced on publication.

## Citation

Citation information will be added on publication.
