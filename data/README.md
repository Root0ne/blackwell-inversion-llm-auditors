# Trial logs

Every file is JSON Lines: one trial per line, one JSON object per trial. The files are
published byte-for-byte as they were written; `SHA256SUMS` lists their digests
(`sha256sum -c SHA256SUMS` from this directory) and `tests/test_data_integrity.py` checks the
digests, the record counts and that no `run_key` repeats inside a file.

## Main data set (7,631 trials)

| File | Trials | Contents |
| --- | ---: | --- |
| `calibration/results_c1_haiku55.jsonl` | 680 | Stage C1: parametric difficulty calibration with Haiku 5.5 (480 Exp 1, 200 Exp 2). Written by an earlier version of the harness, so it lacks the reasoning and truncation fields (see below). |
| `main/results_c2_pilot.jsonl` | 5,026 | Stage C2: nine Anthropic models, 2,064 Exp 1 and 2,962 Exp 2 trials. |
| `main/results_c3_expansion.jsonl` | 1,925 | Stage C3: Google Gemini, Gemma 4 and the causal controls, 508 Exp 1 and 1,417 Exp 2 trials. |

`main/results_c3_expansion.jsonl` is the merge of nine of the ten files in
`main/expansion_parts/`, made by `python -m blackwell_inversion.analysis.main_report
--write-merged`: records are taken in a fixed file order, the first record of each
`run_key` wins, and Gemma seeds beyond the balanced design (`gemma-4-31b-it` seeds >= 5015,
`gemma-4-26b-a4b-it` seeds >= 5010) are dropped. The source files are kept for provenance.

| `main/expansion_parts/` | Records |
| --- | ---: |
| `results_c3_anth_key1.jsonl` | 557 |
| `results_c3_anth_key2.jsonl` | 354 |
| `results_c3_gemini.jsonl` | 744 |
| `results_c3_gemini31_free.jsonl` | 120 |
| `results_c3_gemma.jsonl` | 60 |
| `results_c3_gemma26.jsonl` | 34 |
| `results_c3_gemma26_k1.jsonl` | 26 |
| `results_c3_gemma31_clean_k1.jsonl` | 10 |
| `results_c3_gemma31_controls.jsonl` | 20 |
| `results_c3_gemini3_flash_free.jsonl` | 3 |

The three records of `results_c3_gemini3_flash_free.jsonl` are an interface test with a
model that is not in the paper; they are not merged and not part of the 7,631.

## Follow-up runs (not part of the 7,631)

Three further experiments were run on 23 November 2026, after the main data set was
locked. All use the main Exp 2 condition (Majority-3, 5 turns, medium effort) and seeds
5000-5024, and their records also hold the agent's Python code (`tool_log`) and its queried
inputs (`oracle_history`).

| File | Trials | Contents |
| --- | ---: | --- |
| `followup/results_c4_refutable.jsonl` | 100 | The refutable control on the four models not tested with it in the main set: Haiku 4.5, Sonnet 4.6, Opus 5.5, Fable 5.2. |
| `followup/results_c4_protocols.jsonl` | 1,200 | Three protocols (standard, verify-first instruction, answer check) x four conditions (BB, clean code, hard-guard code, refutable control) x four models (Haiku 5.5, Sonnet 5.5, Gemini 3.5 Flash-Lite, Gemini 3.8 Flash) x 25 seeds. |
| `followup/results_c5_check_tools.jsonl` | 163 | The answer check with tools x two conditions (BB, hard-guard code) x 25 seeds. Haiku 5.5, Sonnet 5.5 and Gemini 3.5 Flash-Lite are complete (150 trials); Gemini 3.8 Flash has 13 of 50 because the provider's prepaid credit ran out, and these 13 records are left out of the analysis. |

Sonnet 4.5 (`claude-sonnet-4-5-20250929`) could not be tested with the refutable control
because the API no longer served it on that date.

In `harness_check` the agent uses four turns; if the check fails it gets one tool-free
revision turn. In `harness_check_tools` the agent uses all five turns; if the check fails,
the ten check queries are appended to `queries.json` and the agent gets three more turns,
the first two with tools. In these records `queries_before_check` is the number of queries
the agent spent before the check, and `queries_used` also counts the ten check queries and
any queries made during revision.

## Field dictionary

Fields common to both experiments:

| Field | Meaning |
| --- | --- |
| `run_key` | Unique key of the trial (experiment, model, reasoning settings, condition labels, seed); format in `blackwell_inversion/schedule.py` |
| `experiment` | `exp1_vbb` or `exp2_blackwell` |
| `model` | Provider model id (`mock-...` in dry runs) |
| `effort`, `thinking_mode`, `thinking_budget` | Reasoning settings sent to the provider (`effort` is `default` when none was sent) |
| `mode` | `BB` or `WB` |
| `seed`, `instance_id`, `n` | Instance seed, instance identifier, input dimension |
| `parse_ok` | Whether the final message contained a valid answer |
| `turns`, `queries_used`, `python_calls` | Model turns, oracle queries and Python executions used |
| `input_tokens`, `output_tokens`, `thinking_tokens`, `cache_read_tokens`, `cache_write_tokens` | Token counts summed over the trial |
| `truncated_turns`, `stop_reasons` | Turns cut at the output token cap, and the stop reason of every turn |
| `cost_usd`, `wall_s` | Estimated cost and wall-clock time of the trial |
| `tool_sequence` | Every tool call as `name:ok` or `name:err` (and `harness_check:revise` when a check failed) |
| `final_excerpt` | First 240 characters of the final message |
| `max_tokens_override` | Present when the per-turn output cap was overridden (8192) |

Experiment 1:

| Field | Meaning |
| --- | --- |
| `family`, `regime`, `substrate`, `track` | Instance family, regime, white-box representation (`blackbox` in BB), Python tool on or off |
| `m`, `k`, `eta` | Number of clauses, clause arity, noise rate of the instance |
| `planted` | 1 if the instance carries the backdoor, 0 if clean |
| `decision_int`, `decision_label`, `correct` | The agent's decision (1/0/null, PLANTED/CLEAN/PARSE_ERROR) and whether it was right |

Experiment 2:

| Field | Meaning |
| --- | --- |
| `task_type`, `variant`, `budget_track`, `q_max` | Task, white-box variant (`wb_inv_code` for BB trials, which share the instance), budget track, query budget |
| `true_support`, `true_signs`, `decoy_support` | The hidden rule and the decoy's variables |
| `pred_support`, `pred_signs` | The agent's answer (empty if invalid) |
| `heldout_acc` | Accuracy of the answer on 100 held-out points (0.50 for an invalid answer) |
| `support_f1`, `exact_support`, `exact_rule` | Overlap with the true variables, and exact matches of the variables and of the rule |
| `matches_decoy` | 1 if the answer is the decoy rule |
| `protocol` | `verify_first`, `harness_check` or `harness_check_tools`; absent under the standard protocol |
| `check_queries`, `check_result` | Size and outcome (`pass`, `mismatch`, `invalid`) of the harness check |
| `first_pred_support`, `first_pred_signs`, `first_parse_ok` | The answer before the check |
| `revision_turns`, `queries_before_check` | `harness_check_tools` only (see above) |
| `tool_log`, `oracle_history` | With `--log-tool-inputs`: per tool call the turn, tool, success, Python code and stdout (truncated) or the number of outputs; every queried input as a 0/1 string with its output |

The C1 log predates the fields `effort`, `thinking_mode`, `thinking_tokens`,
`cache_read_tokens`, `cache_write_tokens`, `truncated_turns`, `stop_reasons` and
`task_type` (all its Exp 2 trials are Majority-3). The main report uses C1 only for its
trial count and cost; the calibration analysis is `reports/legacy/report_c1_haiku55.*`.
