# Pilot Analysis Report — Observer-Relative VBB & Computational Blackwell Inversion

- **Total API spend:** `$5.5313`
- **Exp 1 records:** `800` | **Leak Gate:** `PASS` | **H1 Verdict:** `REFUTED_VBB_BREACHED`
- **Exp 2 records:** `1125` | **H2 Verdict:** `SUPPORTED_MECH1_ONLY`

## Experiment 1: Observer-Relative VBB (`C_kXOR`)

| Model | Effort | Track | Regime | Mode | Substrate | N | Parse | ThinkTok | Adv (95% CI) | Delta_WB | p_holm |
| :--- | :---: | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `claude-haiku-5-5` | medium | tool_python | control_impossible | BB | blackbox | 30 | 1.00 | 572 | +0.000 [-0.204, +0.204] | +0.000 | 1 |
| `claude-haiku-5-5` | medium | tool_python | control_impossible | WB | code_clean | 30 | 1.00 | 1877 | +0.000 [-0.204, +0.204] | +0.000 | 1 |
| `claude-haiku-5-5` | medium | tool_python | control_impossible | WB | code_obf | 30 | 0.93 | 3048 | -0.067 [-0.298, +0.144] | -0.067 | 1 |
| `claude-haiku-5-5` | medium | tool_python | control_impossible | WB | relu_weights | 30 | 1.00 | 1718 | +0.000 [-0.319, +0.319] | +0.000 | 1 |
| `claude-haiku-5-5` | medium | tool_python | transition | BB | blackbox | 30 | 1.00 | 627 | +0.000 [-0.204, +0.204] | +0.000 | - |
| `claude-haiku-5-5` | medium | tool_python | transition | WB | code_clean | 30 | 1.00 | 1716 | +0.867 [+0.548, +0.963] | +0.867 | 1.052e-05 |
| `claude-haiku-5-5` | medium | tool_python | transition | WB | code_obf | 30 | 0.83 | 2868 | +0.667 [+0.344, +0.848] | +0.667 | 0.0008996 |
| `claude-haiku-5-5` | medium | tool_python | transition | WB | relu_weights | 30 | 1.00 | 1402 | +0.133 [-0.190, +0.423] | +0.133 | 1 |
| `claude-haiku-5-5` | medium | tool_python | transparent | BB | blackbox | 30 | 1.00 | 606 | +0.000 [-0.204, +0.204] | +0.000 | - |
| `claude-haiku-5-5` | medium | tool_python | transparent | WB | code_clean | 30 | 1.00 | 1614 | +0.867 [+0.548, +0.963] | +0.867 | 1.052e-05 |
| `claude-haiku-5-5` | medium | tool_python | transparent | WB | code_obf | 30 | 0.97 | 2392 | +0.800 [+0.476, +0.930] | +0.800 | 5.261e-05 |
| `claude-haiku-5-5` | medium | tool_python | transparent | WB | relu_weights | 30 | 1.00 | 1695 | +0.200 [-0.141, +0.484] | +0.200 | 1 |
| `claude-haiku-5-5` | medium | tool_python | vbb_hard | BB | blackbox | 30 | 1.00 | 533 | +0.000 [-0.204, +0.204] | +0.000 | - |
| `claude-haiku-5-5` | medium | tool_python | vbb_hard | WB | code_clean | 30 | 1.00 | 1635 | +0.533 [+0.224, +0.752] | +0.533 | 0.008796 |
| `claude-haiku-5-5` | medium | tool_python | vbb_hard | WB | code_obf | 30 | 1.00 | 2117 | +0.200 [-0.080, +0.459] | +0.200 | 1 |
| `claude-haiku-5-5` | medium | tool_python | vbb_hard | WB | relu_weights | 30 | 1.00 | 1470 | -0.200 [-0.481, +0.134] | -0.200 | 1 |
| `claude-haiku-5-5` | medium | tool_python | control_impossible | BB | blackbox | 30 | 1.00 | 639 | +0.000 [-0.204, +0.204] | +0.000 | 1 |
| `claude-haiku-5-5` | medium | tool_python | control_impossible | WB | code_clean | 30 | 1.00 | 2210 | -0.067 [-0.298, +0.144] | -0.067 | 1 |
| `claude-haiku-5-5` | medium | tool_python | control_impossible | WB | code_obf | 30 | 1.00 | 2665 | +0.000 [-0.204, +0.204] | +0.000 | 1 |
| `claude-haiku-5-5` | medium | tool_python | control_impossible | WB | relu_weights | 30 | 1.00 | 1639 | +0.067 [-0.260, +0.374] | +0.067 | 1 |
| `claude-haiku-5-5` | medium | tool_python | vbb_hard | BB | blackbox | 70 | 1.00 | 556 | +0.000 [-0.099, +0.099] | +0.000 | - |
| `claude-haiku-5-5` | medium | tool_python | vbb_hard | WB | code_clean | 30 | 1.00 | 2084 | -0.067 [-0.298, +0.144] | -0.067 | 1 |
| `claude-haiku-5-5` | medium | tool_python | vbb_hard | WB | code_obf | 30 | 1.00 | 2369 | -0.133 [-0.379, +0.092] | -0.133 | 1 |
| `claude-haiku-5-5` | medium | tool_python | vbb_hard | WB | relu_weights | 70 | 1.00 | 1626 | +0.114 [-0.114, +0.327] | +0.114 | 1 |

## Experiment 2: Computational Blackwell Inversion (`C_inv`)

| Model | Effort | Task | Budget Track | Variant | Pairs | Parse(B/W) | Think(B/W) | BB Acc | WB Acc | Delta_WB (95% CI) | p_holm | Gamma_WB (bits) | Decoy |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `claude-haiku-5-5` | high | majority3 | matched_medium_budget | wb_clean | 25 | 0.92/1.00 | 2493/355 | 0.881 | 1.000 | +0.119 [+0.038, +0.212] | 0.0194 | +1.17 | 0.00 |
| `claude-haiku-5-5` | high | majority3 | matched_medium_budget | wb_decoy_trivial_guard | 25 | 0.92/1.00 | 2493/754 | 0.881 | 1.000 | +0.119 [+0.038, +0.212] | 0.03279 | +1.17 | 0.00 |
| `claude-haiku-5-5` | high | majority3 | matched_medium_budget | wb_inv_code | 25 | 0.92/1.00 | 2493/928 | 0.881 | 0.706 | -0.175 [-0.302, -0.037] | 0.1792 | -1.50 | 0.56 |
| `claude-haiku-5-5` | high | majority3 | matched_medium_budget | wb_inv_relu | 25 | 0.92/0.96 | 2493/774 | 0.881 | 0.872 | -0.009 [-0.147, +0.131] | 1 | -0.21 | 0.04 |
| `claude-haiku-5-5` | high | majority3 | matched_query_free_code | wb_clean | 25 | 0.96/1.00 | 2361/588 | 0.886 | 1.000 | +0.114 [+0.042, +0.193] | 0.006799 | +1.16 | 0.00 |
| `claude-haiku-5-5` | high | majority3 | matched_query_free_code | wb_decoy_trivial_guard | 25 | 0.96/1.00 | 2361/856 | 0.886 | 1.000 | +0.114 [+0.042, +0.193] | 0.0154 | +1.16 | 0.00 |
| `claude-haiku-5-5` | high | majority3 | matched_query_free_code | wb_inv_code | 25 | 0.96/1.00 | 2361/1468 | 0.886 | 1.000 | +0.114 [+0.042, +0.193] | 1 | +1.16 | 0.00 |
| `claude-haiku-5-5` | high | majority3 | matched_query_free_code | wb_inv_relu | 25 | 0.96/1.00 | 2361/1064 | 0.886 | 0.980 | +0.095 [+0.012, +0.180] | 1 | +0.49 | 0.00 |
| `claude-haiku-5-5` | high | majority3 | matched_total_budget | wb_clean | 25 | 0.84/1.00 | 2390/278 | 0.653 | 1.000 | +0.347 [+0.251, +0.438] | 0.0002 | +1.49 | 0.00 |
| `claude-haiku-5-5` | high | majority3 | matched_total_budget | wb_decoy_trivial_guard | 25 | 0.84/1.00 | 2390/657 | 0.653 | 0.876 | +0.223 [+0.098, +0.348] | 0.004399 | +0.13 | 0.20 |
| `claude-haiku-5-5` | high | majority3 | matched_total_budget | wb_inv_code | 25 | 0.84/1.00 | 2390/555 | 0.653 | 0.504 | -0.150 [-0.248, -0.054] | 0.05759 | -6.42 | 1.00 |
| `claude-haiku-5-5` | high | majority3 | matched_total_budget | wb_inv_relu | 25 | 0.84/0.88 | 2390/728 | 0.653 | 0.588 | -0.065 [-0.176, +0.048] | 1 | -2.89 | 0.00 |
| `claude-haiku-5-5` | low | majority3 | matched_medium_budget | wb_clean | 25 | 0.84/1.00 | 1036/104 | 0.878 | 1.000 | +0.122 [+0.040, +0.207] | 0.0178 | +1.03 | 0.00 |
| `claude-haiku-5-5` | low | majority3 | matched_medium_budget | wb_decoy_trivial_guard | 25 | 0.84/1.00 | 1036/534 | 0.878 | 0.936 | +0.057 [-0.050, +0.161] | 0.4257 | -0.05 | 0.08 |
| `claude-haiku-5-5` | low | majority3 | matched_medium_budget | wb_inv_code | 25 | 0.84/1.00 | 1036/554 | 0.878 | 0.920 | +0.042 [-0.085, +0.163] | 1 | -0.12 | 0.16 |
| `claude-haiku-5-5` | low | majority3 | matched_medium_budget | wb_inv_relu | 25 | 0.84/1.00 | 1036/588 | 0.878 | 0.931 | +0.052 [-0.051, +0.158] | 1 | -0.07 | 0.00 |
| `claude-haiku-5-5` | low | majority3 | matched_query_free_code | wb_clean | 25 | 0.88/1.00 | 987/114 | 0.916 | 1.000 | +0.084 [+0.020, +0.167] | 0.06639 | +0.88 | 0.00 |
| `claude-haiku-5-5` | low | majority3 | matched_query_free_code | wb_decoy_trivial_guard | 25 | 0.88/1.00 | 987/448 | 0.916 | 0.984 | +0.068 [-0.011, +0.155] | 0.1302 | +0.26 | 0.00 |
| `claude-haiku-5-5` | low | majority3 | matched_query_free_code | wb_inv_code | 25 | 0.88/1.00 | 987/572 | 0.916 | 0.980 | +0.064 [-0.019, +0.151] | 1 | +0.21 | 0.00 |
| `claude-haiku-5-5` | low | majority3 | matched_query_free_code | wb_inv_relu | 25 | 0.88/0.96 | 987/574 | 0.916 | 0.980 | +0.064 [-0.020, +0.151] | 1 | +0.21 | 0.00 |
| `claude-haiku-5-5` | low | majority3 | matched_total_budget | wb_clean | 25 | 0.80/1.00 | 1028/168 | 0.663 | 1.000 | +0.337 [+0.241, +0.431] | 0.0002 | +0.98 | 0.00 |
| `claude-haiku-5-5` | low | majority3 | matched_total_budget | wb_decoy_trivial_guard | 25 | 0.80/1.00 | 1028/476 | 0.663 | 0.785 | +0.122 [-0.032, +0.278] | 0.1302 | -1.04 | 0.40 |
| `claude-haiku-5-5` | low | majority3 | matched_total_budget | wb_inv_code | 25 | 0.80/1.00 | 1028/424 | 0.663 | 0.522 | -0.141 [-0.241, -0.044] | 0.07499 | -5.48 | 0.96 |
| `claude-haiku-5-5` | low | majority3 | matched_total_budget | wb_inv_relu | 25 | 0.80/0.92 | 1028/645 | 0.663 | 0.576 | -0.086 [-0.183, +0.012] | 0.6071 | -3.68 | 0.00 |
| `claude-haiku-5-5` | medium | majority3 | matched_medium_budget | wb_clean | 25 | 0.96/1.00 | 1409/155 | 0.972 | 1.000 | +0.028 [+0.000, +0.077] | 0.2416 | +0.85 | 0.00 |
| `claude-haiku-5-5` | medium | majority3 | matched_medium_budget | wb_decoy_trivial_guard | 25 | 0.96/1.00 | 1409/700 | 0.972 | 0.978 | +0.006 [-0.059, +0.068] | 1 | +0.15 | 0.04 |
| `claude-haiku-5-5` | medium | majority3 | matched_medium_budget | wb_inv_code | 25 | 0.96/1.00 | 1409/690 | 0.972 | 0.867 | -0.105 [-0.187, -0.034] | 0.2054 | -0.55 | 0.28 |
| `claude-haiku-5-5` | medium | majority3 | matched_medium_budget | wb_inv_relu | 25 | 0.96/1.00 | 1409/742 | 0.972 | 0.991 | +0.019 [-0.019, +0.070] | 1 | +0.48 | 0.00 |
| `claude-haiku-5-5` | medium | majority3 | matched_query_free_code | wb_clean | 25 | 0.80/1.00 | 1594/196 | 0.813 | 1.000 | +0.187 [+0.095, +0.291] | 0.003199 | +1.34 | 0.00 |
| `claude-haiku-5-5` | medium | majority3 | matched_query_free_code | wb_decoy_trivial_guard | 25 | 0.80/1.00 | 1594/681 | 0.813 | 1.000 | +0.187 [+0.095, +0.291] | 0.005799 | +1.34 | 0.00 |
| `claude-haiku-5-5` | medium | majority3 | matched_query_free_code | wb_inv_code | 25 | 0.80/1.00 | 1594/780 | 0.813 | 1.000 | +0.187 [+0.095, +0.291] | 1 | +1.34 | 0.00 |
| `claude-haiku-5-5` | medium | majority3 | matched_query_free_code | wb_inv_relu | 25 | 0.80/1.00 | 1594/691 | 0.813 | 1.000 | +0.187 [+0.095, +0.291] | 1 | +1.34 | 0.00 |
| `claude-haiku-5-5` | medium | majority3 | matched_total_budget | wb_clean | 25 | 0.80/1.00 | 1226/245 | 0.836 | 1.000 | +0.164 [+0.080, +0.259] | 0.002 | +1.03 | 0.00 |
| `claude-haiku-5-5` | medium | majority3 | matched_total_budget | wb_decoy_trivial_guard | 25 | 0.80/1.00 | 1226/541 | 0.836 | 0.880 | +0.044 [-0.077, +0.167] | 0.5053 | -0.31 | 0.24 |
| `claude-haiku-5-5` | medium | majority3 | matched_total_budget | wb_inv_code | 25 | 0.80/1.00 | 1226/498 | 0.836 | 0.566 | -0.270 [-0.371, -0.158] | 0.003599 | -3.85 | 0.88 |
| `claude-haiku-5-5` | medium | majority3 | matched_total_budget | wb_inv_relu | 25 | 0.80/0.88 | 1226/560 | 0.836 | 0.562 | -0.274 [-0.386, -0.152] | 0.006799 | -3.94 | 0.00 |
