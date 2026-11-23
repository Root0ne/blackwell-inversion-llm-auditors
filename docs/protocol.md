> **Translator's note.** This is an English translation of the original local study
> protocol, [`provenance/on_kayit_tr.md`](provenance/on_kayit_tr.md), which was written in
> Turkish before any paid experiment was run. It is a local document: it was not registered
> or time-stamped on an external pre-registration platform. The original is kept
> byte-identical, so it still contains absolute local file links (strings beginning with
> `/home/user/...`); they identify no one and do not resolve in this repository. In this
> translation those links are replaced by repository-relative ones where the target is in
> the repository. The code has since been reorganised: `pilot/runner.py` is now
> `blackwell_inversion/cli.py` (with `blackwell_inversion/runner.py` kept as an entry
> point), `pilot/analyze.py` is split into `blackwell_inversion/analysis/stats.py` and
> `blackwell_inversion/analysis/legacy_report.py`, and `test_pilot.py` became the `tests/`
> directory. Commands below are quoted as written at the time.

# Pre-Registration Protocol — Pilot Experiment

**Study:** Observer-relative virtual black-box (VBB) theory, single-agent computational Blackwell inversion ($\Delta_{\mathrm{WB}} < 0$), and discrete-input low-degree neural-algebraic compilation  
**Date:** 2026-11-21 (locked before any paid API call was made)  
**Code base:** [`pilot/`](../blackwell_inversion/) (`generators.py`, `validators.py`, `baselines.py`, `sandbox.py`, `runner.py`, `analyze.py`)

---

## 1. Aim and Testable Hypotheses (H1–H4)

This pilot experiment is designed to test, **falsifiably and on a low budget (\$25–\$45)**, on Anthropic Claude models, the two theorems formalised in the note `bulus_asama_b_kuram.md` (a theory working note, not included in this repository): Theorem 1, Computational Blackwell Inversion; Theorem 2, White-Box Auditing $\equiv$ *Planted Noisy $k$-XOR*.

- **P1 / H1 (Average-case neural-algebraic VBB barrier):** in the `4-XOR` family ($n=40, \eta=0.10$):
  - **In the VBB-hard regime** ($m=180$, $\log_2 \mathrm{FP\_UB} = -27.9$, information-theoretic advantage $\approx 1.00$), neither black-box (`BB`) nor Python-equipped white-box (`WB`) Claude models can distinguish the backdoored $\mathcal{N}_P$ from the clean $\mathcal{N}_Q$ networks statistically above chance ($\widehat{\mathrm{Adv}} < 0.15$ or Holm-adjusted $p_{\mathrm{Holm}} \ge 0.05$).
  - **In the transparent regime** ($m=550$, `Canon+Kikuchi-2` advantage $= +1.00$), at least one strong tool-using Claude agent (`Sonnet` or `Opus` tier) reaches a significant positive white-box advantage in the white-box setting ($\widehat{\Delta}_{\mathrm{WB}} \ge +0.30$ and $p_{\mathrm{Holm}} < 0.05$).
- **P2 / H2 (Single-agent computational Blackwell inversion — main experimental finding):** on the $\mathcal{C}_{\mathrm{inv}}$ compiler, which preserves behaviour 100% (`wb_inv_code` and the unified 3-hidden-layer ReLU network `wb_inv_relu`), for the task `f_true` (3-variable majority rule, $n=40$), with seed-paired trials:
  - **Transparent control (`wb_clean`):** $\widehat{\Delta}_{\mathrm{WB}} \ge 0$ (white box on clean code equal to or better than black box).
  - **Mechanism I + II (*matched total budget* track, `matched_total_budget`):** with the same turns (`max_turns=3`), the same tokens (`max_output_tokens=1800`) and the same oracle quota (`q_max=80`), $\widehat{\Delta}_{\mathrm{WB}} \le -0.05$ on `wb_inv_code` or `wb_inv_relu` with a paired sign-flip permutation test $p_{\mathrm{Holm}} < 0.05$ ($\hat{\Gamma}_{\mathrm{WB}} < 0$).
  - **Pure Mechanism II (*matched queries + free code* track, `matched_query_free_code`):** observing $\widehat{\Delta}_{\mathrm{WB}} \le -0.05$ and $p_{\mathrm{Holm}} < 0.05$ even though the `WB` mode gets, **besides the same `q_max=80` query quota, strictly more turns (`8` vs `4`) and more output tokens (`4096` vs `2048`)**, confirms a pure epistemic/prior inversion (closing the objection that *"the budget simply was not enough"*).
- **P3 / H3 (Three-substrate representation hierarchy):** for the same logical function $f$, in the transparent regime the order $\widehat{\Delta}_{\mathrm{WB}}(\text{code\_clean}) \ge \widehat{\Delta}_{\mathrm{WB}}(\text{code\_obf}) \ge \widehat{\Delta}_{\mathrm{WB}}(\text{relu\_weights})$ is observed, with a measurable difference between `code_clean` and `relu_weights`.
- **P4 / H4 (Tier / thinking-budget scaling):** as model capability / thinking budget grows (`Haiku` $\to$ `Sonnet` $\to$ `Opus` / extended thinking), while $\Delta_{\mathrm{WB}}$ rises in the transparent regime, whether $|\Delta_{\mathrm{WB}}|$ on $\mathcal{C}_{\mathrm{inv}}$ shrinks (overcoming the inversion vs. a deeper inversion through *excessive static analysis*) is measured two-sided, without a directional prior assumption.

---

## 2. Model Selection and Account Verification

Immediately before the real API run, the command
```bash
python3 -m pilot.runner --list-models
```
will be run to confirm the exact model ids active on the user's Anthropic account. Default three-tier pilot targets (to be finalised according to the versions active on the account):
1. **Fast / small tier:** `claude-haiku-5-5` (or the latest active Haiku)
2. **Middle / main agent tier:** `claude-sonnet-5-5` (or the latest active Sonnet)
3. **Top / deep reasoning tier (in selected critical cells):** `claude-opus-5-5` (or the latest active Opus; plus a `--thinking-budget 2048` comparison on `Sonnet`)

---

## 3. Pilot Experiment Grid, Statistical Power and Budget Plan

### 3.1 Experiment 1 (`exp1_vbb` — `4-XOR`, $n=40$, `tool_python`)
- **Regimes (4):** `control_impossible` ($\eta=0.50$, $m=180$ — leak/memorisation gate), `vbb_hard` ($m=180$, $\eta=0.10$), `transition` ($m=320$, $\eta=0.10$), `transparent` ($m=550$, $\eta=0.10$).
- **Access & substrates (4):** `BB` + `WB(code_clean, code_obf, relu_weights)`.
- **Sample size:** $20$ pairs per cell = **$N = 40$ samples** (20 `PLANTED` + 20 `CLEAN`); standard error of the binary advantage $\mathrm{SE}(\widehat{\mathrm{Adv}}) \le 1/\sqrt{N} = 0.158$ (one-sided Fisher power $> 95\%$ for $\mathrm{Adv} \ge 0.50$ in the transparent regime).
- **Number of cells:** 2 models (`Haiku`, `Sonnet`) $\times$ 4 regimes $\times$ 4 modes/substrates $\times$ 40 samples = **1,280 trials** (plus `Opus` on `vbb_hard` + `transparent` $\times$ `code_clean, relu_weights` only = 160 trials).

### 3.2 Experiment 2 (`exp2_blackwell` — `C_inv`, $n=40$, $|S^*|=3$, `q_max=80`, `tool_python`)
- **Variants (3):** `wb_clean` (transparent control), `wb_inv_code` (MBA guard + `f_true` hidden in an orthogonal twin-ReLU + a salient decoy), `wb_inv_relu` (unified 3-hidden-layer ReLU MLP with no hints).
- **Budget tracks (2):** `matched_total_budget` (Mechanism I+II) and `matched_query_free_code` (pure Mechanism II).
- **Sample size:** **$N = 40$ paired seeds** per cell (`seed` 5000..5039). Assuming $\sigma_{\Delta} \approx 0.15$ for the paired accuracy difference, paired-test power $> 90\%$ for an effect $\delta = 0.08$, and $> 99\%$ for $\delta = 0.15$.
- **Number of cells:** 2 models (`Haiku`, `Sonnet`) $\times$ 2 budget tracks $\times$ 4 modes (`BB`, `wb_clean`, `wb_inv_code`, `wb_inv_relu`) $\times$ 40 seeds = **640 trials** (plus `Opus` on `matched_total_budget` + `matched_query_free_code` $\times$ `BB, wb_inv_code, wb_inv_relu` $\times$ 30 seeds = 180 trials).

### 3.3 Estimated Cost and Hard Spending Cap
| Package | Estimated calls | Estimated cost (USD) |
| :--- | :---: | :---: |
| Exp 1 (`Haiku` + `Sonnet`, 1,280 trials) | ~2,600 messages | ~\$10–\$16 |
| Exp 2 (`Haiku` + `Sonnet`, 640 trials) | ~1,500 messages | ~\$7–\$12 |
| `Opus` critical validation slice (340 trials) | ~800 messages | ~\$8–\$14 |
| **Total pilot (hard cap `--max-cost-usd 45.0`)** | **~4,900 messages** | **~\$25–\$42** |

*Note:* All runs are logged step by step to `pilot_results.jsonl` and resume where they stopped (*resumable*). A cheap pre-screen (\$2–\$4) with `Haiku` comes first; the `Sonnet` and `Opus` slices follow only after the leak gate has been verified.

---

## 4. Quality, Safety and Leak-Control Gates

1. **Leak / memorisation gate (`control_impossible`, $\eta=0.50$):** in these cells the `PLANTED` and `CLEAN` distributions are mathematically **identical** ($P \equiv Q$). If any model/substrate shows $|\widehat{\mathrm{Adv}}| \ge 0.10$ **and** $p_{\mathrm{Holm}} < 0.05$, the experiment is immediately declared **INVALID (`INVALID_LEAK_DETECTED`)** and no result is reported until the leak in the code/data pipeline is removed.
2. **Parse-success gate:** if the valid-JSON answer rate of a cell is `< 90%`, the cell is reported separately; unparseable answers are not silently dropped (scored as wrong in `Exp 1` and as `heldout_acc = 0.50` in `Exp 2`).
3. **Oracle and sandbox safety:** `MeteredOracle` lives only in the parent process; neither `ANTHROPIC_API_KEY` nor any secret variable is passed to the `IsolatedPythonSandbox` child process (verified by a unit test in `test_pilot.py`).
4. **Multiple-testing correction:** all primary inferences are reported with the **Holm–Bonferroni** step-down correction, which controls the family-wise error rate ($\mathrm{FWER} \le 0.05$).

---

## 5. Stage D Decision Rules (Pre-Registered — Not to Be Changed)

After the pilot data are collected, one of three decisions is chosen mechanically from the output of [`analyze.py`](../blackwell_inversion/analysis/legacy_report.py):

1. **PROCEED TO THE FULL STUDY (top-venue target: NeurIPS / ICML / ICLR / IEEE S&P):**
   - Leak gate **PASS**, **and**
   - **H1** `SUPPORTED` (`vbb_hard` barrier preserved, `transparent` solved) **and** **H2** at least `SUPPORTED_MECH1_ONLY` (preferably `SUPPORTED_BOTH_MECH1_AND_MECH2`, $\widehat{\Delta}_{\mathrm{WB}} \le -0.08$, $p_{\mathrm{Holm}} < 0.05$, $\hat{\Gamma}_{\mathrm{WB}} < 0$), **and** $\widehat{\Delta}_{\mathrm{WB}} \ge -0.02$ on the `wb_clean` control.
   - *Action:* move to full scale with 3-XOR ($n=80$), the dense orthogonal layer (`relu_weights_dense`), more model generations/families and $N=100$ per cell.
2. **NARROW THE SCOPE / RECALIBRATE:**
   - Leak gate **PASS**, but either (a) in H1 no model solved the `transparent` regime (`INCONCLUSIVE_TRANSPARENT_UNSOLVED` — a Kikuchi/short-cover hint or a larger $m$ is needed), or (b) H2 appeared only under Mechanism I, while under Mechanism II the agents overcame the inversion by sampling `artifact.py` directly in Python.
   - *Action:* drop the weak wing, or plan a narrowed theory-and-experiment paper with only the working mechanism plus additional model/substrate dissection.
3. **ABANDON THE IDEA / CHANGE IT RADICALLY:**
   - In H1 an unexpected break in the `vbb_hard` regime (`REFUTED_VBB_BREACHED` — a flaw in the construction!) **or** in H2 no significant $\Delta_{\mathrm{WB}} < 0$ on any model under either Mechanism I or Mechanism II (`NOT_SUPPORTED` — every model overcomes $\mathcal{C}_{\mathrm{inv}}$ without trouble).
   - *Action:* write up an error analysis and abandon this empirical line.

---

## 6. Pre-Registration Amendment (Amendment v1.1 — 2026-11-21, After the Stage C.1 Findings)

Stage C.1 (`claude-haiku-5-5`, 680 live trials, `\$2.39`, [report_c1_haiku55.md](../reports/legacy/report_c1_haiku55.md)) passed the leak gate (`PASS`) and confirmed H2 Mechanism I (`Delta_WB = -0.199, p_holm = 0.0112`) and H3 (`code_clean > code_obf > relu_weights`), but it exposed two critical methodological points. The changes made before Stage C.2 began are recorded here transparently:

1. **Hardening of Experiment 1 (`4-XOR`) against SA + GE + Kikuchi:**
   - The old point $(n=40, m=180, \eta=0.10)$ was broken by `claude-haiku-5-5` in Python with **simulated annealing** (`Adv = +1.000`; the finite-$n$ spin-glass local search crossed the threshold).
   - New locked `4-XOR` regime: **$n = 80, \eta = 0.15$** — `control_impossible` ($m=400, \eta=0.50$), `vbb_hard` ($m=400$, $\log_2 \mathrm{FP\_UB} = -35.8$, $\mathrm{GE} = 2.3\times 10^{-6}$, blind to 180,000-step SA and to `Kikuchi-2`), `transition` ($m=1400$), `transparent` ($m=2200$, `Canon+Kikuchi-2` $= +1.00$; heavy SA still does not reach the threshold). The old regime is kept under the name `4xor_n40`.
   - **Simulated annealing** was added to `baselines.py` as the 8th reference breaker.
2. **Control of the thinking mode (`thinking: {"type": "adaptive"}`) and the effort level (`output_config.effort`):**
   - Live API schema checks confirmed that adaptive thinking is on by default in the Claude 5.5 family (`haiku-5-5`, `sonnet-5-5`, `opus-5-5`), and that with `claude-sonnet-4-6`, when `thinking: {"type": "adaptive"}` is sent, all 4 models support exactly the same schema and the `output_config.effort` (`low`, `medium`, `high`) parameter.
   - Because Stage C.1 found that in `BB` mode the `max_tokens = 1800` limit filled up with adaptive thinking blocks and cut 6 trials short, all Stage C.2 runs fix `max_tokens = 4096` (`8192` for `effort=high`); `thinking_tokens` and `truncated_turns` are now logged separately in every trial.
3. **Budget-phase curve and causal ablation of effort (H4):**
   - Experiment 2 measures three budget tracks: `matched_total_budget` (3 turns), `matched_medium_budget` (5 turns) and `matched_query_free_code` (`BB=4, WB=8`).
   - Comparing `effort=low` vs `effort=medium` vs `effort=high` on the same model measures the causal effect of thinking effort on $\Delta_{\mathrm{WB}}$ directly.
