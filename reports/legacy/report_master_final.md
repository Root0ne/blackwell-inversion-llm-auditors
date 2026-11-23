# Pilot Analysis Report — Observer-Relative VBB & Computational Blackwell Inversion

- **Total logged JSONL API spend:** `$176.4018`
- **Exp 1 records:** `2064` | **Leak Gate:** `PASS` | **H1 Verdict (pooled):** `REFUTED_VBB_BREACHED`
- **H1 Verdict by Family:** `3xor=REFUTED_VBB_BREACHED`, `4xor=REFUTED_VBB_BREACHED`, `4xor_strict=SUPPORTED`
- **Exp 2 records:** `2962` | **H2 Verdict:** `SUPPORTED_MECH1_ONLY`

## Experiment 1: Observer-Relative VBB (`C_kXOR`)

| Model | Effort | Family | Track | Regime | Mode | Substrate | N | Parse | ThinkTok | Adv (95% CI) | Delta_WB | p_raw | p_holm_fam |
| :--- | :---: | :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `claude-fable-5-2` | medium | `4xor_strict` | tool_python | transparent | BB | blackbox | 24 | 1.00 | 164 | +0.000 [-0.242, +0.242] | +0.000 | - | - |
| `claude-fable-5-2` | medium | `4xor_strict` | tool_python | transparent | WB | code_clean | 24 | 1.00 | 1177 | +1.000 [+0.657, +1.000] | +1.000 | 3.698e-07 | 1.072e-05 |
| `claude-fable-5-2` | medium | `4xor_strict` | tool_python | transparent | WB | relu_weights | 24 | 1.00 | 1506 | +1.000 [+0.657, +1.000] | +1.000 | 3.698e-07 | 1.072e-05 |
| `claude-fable-5-2` | medium | `4xor_strict` | tool_python | transparent | WB | relu_weights_dense | 22 | 1.00 | 1814 | +0.636 [+0.219, +0.821] | +0.636 | 0.004461 | 0.1115 |
| `claude-fable-5-2` | medium | `4xor_strict` | tool_python | vbb_hard | BB | blackbox | 24 | 1.00 | 130 | +0.000 [-0.242, +0.242] | +0.000 | - | - |
| `claude-fable-5-2` | medium | `4xor_strict` | tool_python | vbb_hard | WB | code_clean | 24 | 0.83 | 2986 | +0.000 [-0.242, +0.242] | +0.000 | 1 | 1 |
| `claude-fable-5-2` | medium | `4xor_strict` | tool_python | vbb_hard | WB | relu_weights | 24 | 0.92 | 2849 | +0.000 [-0.242, +0.242] | +0.000 | 1 | 1 |
| `claude-fable-5-2` | medium | `4xor_strict` | tool_python | vbb_hard | WB | relu_weights_dense | 24 | 1.00 | 1852 | +0.250 [-0.109, +0.540] | +0.250 | 0.1854 | 1 |
| `claude-haiku-5-5` | medium | `3xor` | tool_python | control_impossible | BB | blackbox | 24 | 1.00 | 512 | +0.000 [-0.242, +0.242] | +0.000 | 1 | 1 |
| `claude-haiku-5-5` | medium | `3xor` | tool_python | control_impossible | WB | code_clean | 24 | 1.00 | 2103 | +0.000 [-0.242, +0.242] | +0.000 | 1 | 1 |
| `claude-haiku-5-5` | medium | `3xor` | tool_python | control_impossible | WB | code_obf | 24 | 1.00 | 2394 | +0.000 [-0.242, +0.242] | +0.000 | 1 | 1 |
| `claude-haiku-5-5` | medium | `3xor` | tool_python | control_impossible | WB | relu_weights | 24 | 1.00 | 1908 | +0.167 [-0.203, +0.481] | +0.167 | 0.6802 | 1 |
| `claude-haiku-5-5` | medium | `3xor` | tool_python | control_impossible | WB | relu_weights_dense | 24 | 1.00 | 961 | -0.083 [-0.354, +0.169] | -0.083 | 1 | 1 |
| `claude-haiku-5-5` | medium | `3xor` | tool_python | transparent | BB | blackbox | 24 | 1.00 | 512 | +0.000 [-0.242, +0.242] | +0.000 | - | - |
| `claude-haiku-5-5` | medium | `3xor` | tool_python | transparent | WB | code_clean | 24 | 1.00 | 1934 | +0.917 [+0.553, +0.985] | +0.917 | 4.807e-06 | 3.846e-05 |
| `claude-haiku-5-5` | medium | `3xor` | tool_python | transparent | WB | code_obf | 24 | 1.00 | 2198 | +0.750 [+0.378, +0.911] | +0.750 | 0.0001683 | 0.001178 |
| `claude-haiku-5-5` | medium | `3xor` | tool_python | transparent | WB | relu_weights | 24 | 1.00 | 1488 | -0.167 [-0.481, +0.203] | -0.167 | 0.8931 | 1 |
| `claude-haiku-5-5` | medium | `3xor` | tool_python | transparent | WB | relu_weights_dense | 24 | 1.00 | 1098 | -0.167 [-0.448, +0.104] | -0.167 | 1 | 1 |
| `claude-haiku-5-5` | medium | `3xor` | tool_python | vbb_hard | BB | blackbox | 24 | 1.00 | 502 | +0.000 [-0.242, +0.242] | +0.000 | - | - |
| `claude-haiku-5-5` | medium | `3xor` | tool_python | vbb_hard | WB | code_clean | 24 | 1.00 | 1787 | +0.500 [+0.122, +0.734] | +0.500 | 0.0136 | 0.06798 |
| `claude-haiku-5-5` | medium | `3xor` | tool_python | vbb_hard | WB | code_obf | 24 | 1.00 | 1995 | +0.500 [+0.154, +0.746] | +0.500 | 0.006865 | 0.04119 |
| `claude-haiku-5-5` | medium | `3xor` | tool_python | vbb_hard | WB | relu_weights | 24 | 1.00 | 1796 | +0.250 [-0.125, +0.544] | +0.250 | 0.2002 | 0.8006 |
| `claude-haiku-5-5` | medium | `3xor` | tool_python | vbb_hard | WB | relu_weights_dense | 24 | 1.00 | 859 | +0.083 [-0.169, +0.354] | +0.083 | 0.5 | 1 |
| `claude-haiku-5-5` | medium | `4xor` | tool_python | control_impossible | BB | blackbox | 30 | 1.00 | 572 | +0.000 [-0.204, +0.204] | +0.000 | 1 | 1 |
| `claude-haiku-5-5` | medium | `4xor` | tool_python | control_impossible | WB | code_clean | 30 | 1.00 | 1877 | +0.000 [-0.204, +0.204] | +0.000 | 1 | 1 |
| `claude-haiku-5-5` | medium | `4xor` | tool_python | control_impossible | WB | code_obf | 30 | 0.93 | 3048 | -0.067 [-0.298, +0.144] | -0.067 | 1 | 1 |
| `claude-haiku-5-5` | medium | `4xor` | tool_python | control_impossible | WB | relu_weights | 30 | 1.00 | 1718 | +0.000 [-0.319, +0.319] | +0.000 | 1 | 1 |
| `claude-haiku-5-5` | medium | `4xor` | tool_python | transition | BB | blackbox | 30 | 1.00 | 627 | +0.000 [-0.204, +0.204] | +0.000 | - | - |
| `claude-haiku-5-5` | medium | `4xor` | tool_python | transition | WB | code_clean | 30 | 1.00 | 1716 | +0.867 [+0.548, +0.963] | +0.867 | 8.768e-07 | 8.768e-06 |
| `claude-haiku-5-5` | medium | `4xor` | tool_python | transition | WB | code_obf | 30 | 0.83 | 2868 | +0.667 [+0.344, +0.848] | +0.667 | 9.995e-05 | 0.0006997 |
| `claude-haiku-5-5` | medium | `4xor` | tool_python | transition | WB | relu_weights | 30 | 1.00 | 1402 | +0.133 [-0.190, +0.423] | +0.133 | 0.3499 | 0.9243 |
| `claude-haiku-5-5` | medium | `4xor` | tool_python | transparent | BB | blackbox | 30 | 1.00 | 606 | +0.000 [-0.204, +0.204] | +0.000 | - | - |
| `claude-haiku-5-5` | medium | `4xor` | tool_python | transparent | WB | code_clean | 30 | 1.00 | 1614 | +0.867 [+0.548, +0.963] | +0.867 | 8.768e-07 | 8.768e-06 |
| `claude-haiku-5-5` | medium | `4xor` | tool_python | transparent | WB | code_obf | 30 | 0.97 | 2392 | +0.800 [+0.476, +0.930] | +0.800 | 5.261e-06 | 4.208e-05 |
| `claude-haiku-5-5` | medium | `4xor` | tool_python | transparent | WB | relu_weights | 30 | 1.00 | 1695 | +0.200 [-0.141, +0.484] | +0.200 | 0.2311 | 0.9243 |
| `claude-haiku-5-5` | medium | `4xor` | tool_python | transparent | WB | relu_weights_dense | 30 | 1.00 | 1049 | +0.133 [-0.092, +0.379] | +0.133 | 0.2414 | 0.9243 |
| `claude-haiku-5-5` | medium | `4xor` | tool_python | vbb_hard | BB | blackbox | 30 | 1.00 | 533 | +0.000 [-0.204, +0.204] | +0.000 | - | - |
| `claude-haiku-5-5` | medium | `4xor` | tool_python | vbb_hard | WB | code_clean | 30 | 1.00 | 1635 | +0.533 [+0.224, +0.752] | +0.533 | 0.001099 | 0.006597 |
| `claude-haiku-5-5` | medium | `4xor` | tool_python | vbb_hard | WB | code_obf | 30 | 1.00 | 2117 | +0.200 [-0.080, +0.459] | +0.200 | 0.1648 | 0.8238 |
| `claude-haiku-5-5` | medium | `4xor` | tool_python | vbb_hard | WB | relu_weights | 30 | 1.00 | 1470 | -0.200 [-0.481, +0.134] | -0.200 | 0.9359 | 0.9359 |
| `claude-haiku-5-5` | medium | `4xor_strict` | tool_free | transparent | BB | blackbox | 24 | 1.00 | 395 | +0.000 [-0.242, +0.242] | +0.000 | - | - |
| `claude-haiku-5-5` | medium | `4xor_strict` | tool_free | transparent | WB | code_clean | 24 | 1.00 | 628 | +0.000 [-0.242, +0.242] | +0.000 | 1 | 1 |
| `claude-haiku-5-5` | medium | `4xor_strict` | tool_free | transparent | WB | code_obf | 24 | 1.00 | 600 | +0.000 [-0.242, +0.242] | +0.000 | 1 | 1 |
| `claude-haiku-5-5` | medium | `4xor_strict` | tool_free | transparent | WB | relu_weights | 24 | 1.00 | 434 | +0.000 [-0.242, +0.242] | +0.000 | 1 | 1 |
| `claude-haiku-5-5` | medium | `4xor_strict` | tool_free | vbb_hard | BB | blackbox | 24 | 1.00 | 423 | +0.000 [-0.242, +0.242] | +0.000 | - | - |
| `claude-haiku-5-5` | medium | `4xor_strict` | tool_free | vbb_hard | WB | code_clean | 24 | 1.00 | 701 | +0.000 [-0.242, +0.242] | +0.000 | 1 | 1 |
| `claude-haiku-5-5` | medium | `4xor_strict` | tool_free | vbb_hard | WB | code_obf | 24 | 1.00 | 674 | +0.000 [-0.242, +0.242] | +0.000 | 1 | 1 |
| `claude-haiku-5-5` | medium | `4xor_strict` | tool_free | vbb_hard | WB | relu_weights | 24 | 0.96 | 407 | +0.000 [-0.242, +0.242] | +0.000 | 1 | 1 |
| `claude-haiku-5-5` | medium | `4xor_strict` | tool_python | control_impossible | BB | blackbox | 30 | 1.00 | 639 | +0.000 [-0.204, +0.204] | +0.000 | 1 | 1 |
| `claude-haiku-5-5` | medium | `4xor_strict` | tool_python | control_impossible | WB | code_clean | 30 | 1.00 | 2210 | -0.067 [-0.298, +0.144] | -0.067 | 1 | 1 |
| `claude-haiku-5-5` | medium | `4xor_strict` | tool_python | control_impossible | WB | code_obf | 30 | 1.00 | 2665 | +0.000 [-0.204, +0.204] | +0.000 | 1 | 1 |
| `claude-haiku-5-5` | medium | `4xor_strict` | tool_python | control_impossible | WB | relu_weights | 30 | 1.00 | 1639 | +0.067 [-0.260, +0.374] | +0.067 | 1 | 1 |
| `claude-haiku-5-5` | medium | `4xor_strict` | tool_python | vbb_hard | BB | blackbox | 70 | 1.00 | 556 | +0.000 [-0.099, +0.099] | +0.000 | - | - |
| `claude-haiku-5-5` | medium | `4xor_strict` | tool_python | vbb_hard | WB | code_clean | 30 | 1.00 | 2084 | -0.067 [-0.298, +0.144] | -0.067 | 1 | 1 |
| `claude-haiku-5-5` | medium | `4xor_strict` | tool_python | vbb_hard | WB | code_obf | 30 | 1.00 | 2369 | -0.133 [-0.379, +0.092] | -0.133 | 1 | 1 |
| `claude-haiku-5-5` | medium | `4xor_strict` | tool_python | vbb_hard | WB | relu_weights | 70 | 1.00 | 1626 | +0.114 [-0.114, +0.327] | +0.114 | 0.236 | 1 |
| `claude-opus-5-5` | medium | `4xor_strict` | tool_python | transparent | BB | blackbox | 24 | 1.00 | 181 | +0.000 [-0.279, +0.279] | +0.000 | - | - |
| `claude-opus-5-5` | medium | `4xor_strict` | tool_python | transparent | WB | code_clean | 24 | 1.00 | 545 | +0.583 [+0.185, +0.784] | +0.583 | 0.006139 | 0.1473 |
| `claude-opus-5-5` | medium | `4xor_strict` | tool_python | transparent | WB | relu_weights | 24 | 1.00 | 870 | +0.083 [-0.260, +0.403] | +0.083 | 0.5 | 1 |
| `claude-opus-5-5` | medium | `4xor_strict` | tool_python | transparent | WB | relu_weights_dense | 24 | 1.00 | 916 | -0.083 [-0.354, +0.169] | -0.083 | 1 | 1 |
| `claude-opus-5-5` | medium | `4xor_strict` | tool_python | vbb_hard | BB | blackbox | 24 | 1.00 | 171 | -0.167 [-0.448, +0.104] | +0.000 | - | - |
| `claude-opus-5-5` | medium | `4xor_strict` | tool_python | vbb_hard | WB | code_clean | 24 | 1.00 | 894 | -0.083 [-0.354, +0.169] | +0.083 | 1 | 1 |
| `claude-opus-5-5` | medium | `4xor_strict` | tool_python | vbb_hard | WB | relu_weights | 24 | 1.00 | 964 | -0.250 [-0.532, +0.041] | -0.083 | 1 | 1 |
| `claude-opus-5-5` | medium | `4xor_strict` | tool_python | vbb_hard | WB | relu_weights_dense | 24 | 1.00 | 868 | -0.083 [-0.354, +0.169] | +0.083 | 1 | 1 |
| `claude-sonnet-5-5` | medium | `4xor_strict` | tool_python | control_impossible | BB | blackbox | 20 | 1.00 | 0 | +0.000 [-0.278, +0.278] | +0.000 | 1 | 1 |
| `claude-sonnet-5-5` | medium | `4xor_strict` | tool_python | control_impossible | WB | code_clean | 20 | 1.00 | 545 | -0.200 [-0.510, +0.112] | -0.200 | 0.4737 | 1 |
| `claude-sonnet-5-5` | medium | `4xor_strict` | tool_python | control_impossible | WB | relu_weights | 20 | 1.00 | 539 | -0.100 [-0.421, +0.236] | -0.100 | 1 | 1 |
| `claude-sonnet-5-5` | medium | `4xor_strict` | tool_python | transparent | BB | blackbox | 24 | 1.00 | 2 | +0.000 [-0.242, +0.242] | +0.000 | - | - |
| `claude-sonnet-5-5` | medium | `4xor_strict` | tool_python | transparent | WB | code_clean | 24 | 1.00 | 359 | +0.833 [+0.462, +0.953] | +0.833 | 3.365e-05 | 0.0009086 |
| `claude-sonnet-5-5` | medium | `4xor_strict` | tool_python | transparent | WB | code_obf | 24 | 1.00 | 499 | +0.667 [+0.299, +0.862] | +0.667 | 0.000673 | 0.0175 |
| `claude-sonnet-5-5` | medium | `4xor_strict` | tool_python | transparent | WB | relu_weights | 24 | 1.00 | 638 | +0.500 [+0.154, +0.746] | +0.500 | 0.006865 | 0.1579 |
| `claude-sonnet-5-5` | medium | `4xor_strict` | tool_python | transparent | WB | relu_weights_dense | 24 | 1.00 | 446 | -0.167 [-0.457, +0.148] | -0.167 | 0.9534 | 1 |
| `claude-sonnet-5-5` | medium | `4xor_strict` | tool_python | vbb_hard | BB | blackbox | 24 | 1.00 | 0 | +0.000 [-0.242, +0.242] | +0.000 | - | - |
| `claude-sonnet-5-5` | medium | `4xor_strict` | tool_python | vbb_hard | WB | code_clean | 24 | 1.00 | 499 | -0.167 [-0.448, +0.104] | -0.167 | 1 | 1 |
| `claude-sonnet-5-5` | medium | `4xor_strict` | tool_python | vbb_hard | WB | code_obf | 24 | 1.00 | 702 | +0.167 [-0.104, +0.448] | +0.167 | 0.2391 | 1 |
| `claude-sonnet-5-5` | medium | `4xor_strict` | tool_python | vbb_hard | WB | relu_weights | 24 | 1.00 | 550 | +0.083 [-0.169, +0.354] | +0.083 | 0.5 | 1 |
| `claude-sonnet-5-5` | medium | `4xor_strict` | tool_python | vbb_hard | WB | relu_weights_dense | 24 | 1.00 | 358 | -0.083 [-0.354, +0.169] | -0.083 | 1 | 1 |

## Experiment 2: Computational Blackwell Inversion (`C_inv`)

| Model | Effort | Task | Budget Track | Variant | Pairs | Turns(B/W) | Parse(B/W) | Think(B/W) | BB Acc | WB Acc | Delta_WB (95% CI) | p_raw | p_holm_cell | p_holm_all | Gamma_WB (bits) | Decoy |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `claude-fable-5-2` | medium | majority3 | matched_medium_budget | wb_clean | 25 | 3.72/4.64 | 1.00/1.00 | 1133/78 | 0.966 | 1.000 | +0.034 [+0.000, +0.086] | 0.2428 | 0.2428 | 0.2428 | +1.15 | 0.00 |
| `claude-fable-5-2` | medium | majority3 | matched_medium_budget | wb_decoy_trivial_guard | 25 | 3.72/4.00 | 1.00/1.00 | 1133/117 | 0.966 | 1.000 | +0.034 [+0.000, +0.086] | 0.4933 | 0.4933 | 0.4933 | +1.15 | 0.00 |
| `claude-fable-5-2` | medium | majority3 | matched_medium_budget | wb_inv_code | 25 | 3.72/4.12 | 1.00/1.00 | 1133/154 | 0.966 | 1.000 | +0.034 [+0.000, +0.086] | 1 | 1 | 1 | +1.15 | 0.00 |
| `claude-fable-5-2` | medium | majority3 | matched_medium_budget | wb_inv_relu | 25 | 3.72/4.64 | 1.00/1.00 | 1133/132 | 0.966 | 1.000 | +0.034 [+0.000, +0.086] | 1 | 1 | 1 | +1.15 | 0.00 |
| `claude-haiku-4-5-20251001` | default | majority3 | matched_medium_budget | wb_clean | 25 | 5.00/5.00 | 0.36/1.00 | 0/0 | 0.672 | 1.000 | +0.328 [+0.236, +0.420] | 0.0002 | 0.0002 | 0.0002 | +0.58 | 0.00 |
| `claude-haiku-4-5-20251001` | default | majority3 | matched_medium_budget | wb_decoy_trivial_guard | 25 | 5.00/5.00 | 0.36/0.68 | 0/0 | 0.672 | 0.506 | -0.166 [-0.267, -0.071] | 0.004399 | 0.004399 | 0.004399 | -7.32 | 0.68 |
| `claude-haiku-4-5-20251001` | default | majority3 | matched_medium_budget | wb_inv_code | 25 | 5.00/5.00 | 0.36/0.44 | 0/0 | 0.672 | 0.508 | -0.164 [-0.263, -0.071] | 0.002599 | 0.003999 | 0.1066 | -7.32 | 0.44 |
| `claude-haiku-4-5-20251001` | default | majority3 | matched_medium_budget | wb_inv_relu | 25 | 5.00/5.00 | 0.36/0.40 | 0/0 | 0.672 | 0.506 | -0.166 [-0.265, -0.070] | 0.002 | 0.003999 | 0.08398 | -7.32 | 0.00 |
| `claude-haiku-5-5` | high | majority3 | matched_medium_budget | wb_clean | 25 | 4.76/4.52 | 0.92/1.00 | 2493/355 | 0.881 | 1.000 | +0.119 [+0.038, +0.212] | 0.0194 | 0.0194 | 0.0194 | +1.17 | 0.00 |
| `claude-haiku-5-5` | high | majority3 | matched_medium_budget | wb_decoy_trivial_guard | 25 | 4.76/5.00 | 0.92/1.00 | 2493/754 | 0.881 | 1.000 | +0.119 [+0.038, +0.212] | 0.03279 | 0.03279 | 0.03279 | +1.17 | 0.00 |
| `claude-haiku-5-5` | high | majority3 | matched_medium_budget | wb_inv_code | 25 | 4.76/5.00 | 0.92/1.00 | 2493/928 | 0.881 | 0.706 | -0.175 [-0.302, -0.037] | 0.0128 | 0.02559 | 0.4735 | -1.54 | 0.56 |
| `claude-haiku-5-5` | high | majority3 | matched_medium_budget | wb_inv_relu | 25 | 4.76/5.00 | 0.92/0.96 | 2493/774 | 0.881 | 0.872 | -0.009 [-0.147, +0.131] | 0.4431 | 0.4431 | 1 | -0.30 | 0.04 |
| `claude-haiku-5-5` | high | majority3 | matched_query_free_code | wb_clean | 25 | 3.92/5.20 | 0.96/1.00 | 2361/588 | 0.886 | 1.000 | +0.114 [+0.042, +0.193] | 0.006799 | 0.006799 | 0.006799 | +1.16 | 0.00 |
| `claude-haiku-5-5` | high | majority3 | matched_query_free_code | wb_decoy_trivial_guard | 25 | 3.92/6.60 | 0.96/1.00 | 2361/856 | 0.886 | 1.000 | +0.114 [+0.042, +0.193] | 0.0154 | 0.0154 | 0.0154 | +1.16 | 0.00 |
| `claude-haiku-5-5` | high | majority3 | matched_query_free_code | wb_inv_code | 25 | 3.92/7.52 | 0.96/1.00 | 2361/1468 | 0.886 | 1.000 | +0.114 [+0.042, +0.193] | 1 | 1 | 1 | +1.16 | 0.00 |
| `claude-haiku-5-5` | high | majority3 | matched_query_free_code | wb_inv_relu | 25 | 3.92/7.28 | 0.96/1.00 | 2361/1064 | 0.886 | 0.980 | +0.095 [+0.012, +0.180] | 0.9874 | 1 | 1 | +0.55 | 0.00 |
| `claude-haiku-5-5` | high | majority3 | matched_total_budget | wb_clean | 25 | 3.00/3.00 | 0.84/1.00 | 2390/278 | 0.653 | 1.000 | +0.347 [+0.251, +0.438] | 0.0002 | 0.0002 | 0.0002 | +1.49 | 0.00 |
| `claude-haiku-5-5` | high | majority3 | matched_total_budget | wb_decoy_trivial_guard | 25 | 3.00/3.00 | 0.84/1.00 | 2390/657 | 0.653 | 0.876 | +0.223 [+0.098, +0.348] | 0.004399 | 0.004399 | 0.004399 | +0.05 | 0.20 |
| `claude-haiku-5-5` | high | majority3 | matched_total_budget | wb_inv_code | 25 | 3.00/3.00 | 0.84/1.00 | 2390/555 | 0.653 | 0.504 | -0.150 [-0.248, -0.054] | 0.003599 | 0.007199 | 0.144 | -6.42 | 1.00 |
| `claude-haiku-5-5` | high | majority3 | matched_total_budget | wb_inv_relu | 25 | 3.00/3.00 | 0.84/0.88 | 2390/728 | 0.653 | 0.588 | -0.065 [-0.176, +0.048] | 0.1384 | 0.1384 | 1 | -3.01 | 0.00 |
| `claude-haiku-5-5` | low | majority3 | matched_medium_budget | wb_clean | 25 | 3.56/4.68 | 0.84/1.00 | 1036/104 | 0.878 | 1.000 | +0.122 [+0.040, +0.207] | 0.0178 | 0.0178 | 0.0178 | +1.03 | 0.00 |
| `claude-haiku-5-5` | low | majority3 | matched_medium_budget | wb_decoy_trivial_guard | 25 | 3.56/4.40 | 0.84/1.00 | 1036/534 | 0.878 | 0.936 | +0.057 [-0.050, +0.161] | 0.4257 | 0.4257 | 0.4257 | +0.02 | 0.08 |
| `claude-haiku-5-5` | low | majority3 | matched_medium_budget | wb_inv_code | 25 | 3.56/4.76 | 0.84/1.00 | 1036/554 | 0.878 | 0.920 | +0.042 [-0.085, +0.163] | 0.7612 | 1 | 1 | -0.08 | 0.16 |
| `claude-haiku-5-5` | low | majority3 | matched_medium_budget | wb_inv_relu | 25 | 3.56/4.44 | 0.84/1.00 | 1036/588 | 0.878 | 0.931 | +0.052 [-0.051, +0.158] | 0.8458 | 1 | 1 | -0.01 | 0.00 |
| `claude-haiku-5-5` | low | majority3 | matched_query_free_code | wb_clean | 25 | 3.32/4.84 | 0.88/1.00 | 987/114 | 0.916 | 1.000 | +0.084 [+0.020, +0.167] | 0.06639 | 0.06639 | 0.06639 | +0.88 | 0.00 |
| `claude-haiku-5-5` | low | majority3 | matched_query_free_code | wb_decoy_trivial_guard | 25 | 3.32/4.92 | 0.88/1.00 | 987/448 | 0.916 | 0.984 | +0.068 [-0.011, +0.155] | 0.1302 | 0.1302 | 0.1302 | +0.32 | 0.00 |
| `claude-haiku-5-5` | low | majority3 | matched_query_free_code | wb_inv_code | 25 | 3.32/5.60 | 0.88/1.00 | 987/572 | 0.916 | 0.980 | +0.064 [-0.019, +0.151] | 0.969 | 1 | 1 | +0.27 | 0.00 |
| `claude-haiku-5-5` | low | majority3 | matched_query_free_code | wb_inv_relu | 25 | 3.32/5.20 | 0.88/0.96 | 987/574 | 0.916 | 0.980 | +0.064 [-0.020, +0.151] | 0.9676 | 1 | 1 | +0.27 | 0.00 |
| `claude-haiku-5-5` | low | majority3 | matched_total_budget | wb_clean | 25 | 2.84/3.00 | 0.80/1.00 | 1028/168 | 0.663 | 1.000 | +0.337 [+0.241, +0.431] | 0.0002 | 0.0002 | 0.0002 | +0.98 | 0.00 |
| `claude-haiku-5-5` | low | majority3 | matched_total_budget | wb_decoy_trivial_guard | 25 | 2.84/3.00 | 0.80/1.00 | 1028/476 | 0.663 | 0.785 | +0.122 [-0.032, +0.278] | 0.1302 | 0.1302 | 0.1302 | -1.16 | 0.40 |
| `claude-haiku-5-5` | low | majority3 | matched_total_budget | wb_inv_code | 25 | 2.84/3.00 | 0.80/1.00 | 1028/424 | 0.663 | 0.522 | -0.141 [-0.241, -0.044] | 0.004999 | 0.009998 | 0.195 | -5.56 | 0.96 |
| `claude-haiku-5-5` | low | majority3 | matched_total_budget | wb_inv_relu | 25 | 2.84/3.00 | 0.80/0.92 | 1028/645 | 0.663 | 0.576 | -0.086 [-0.183, +0.012] | 0.05059 | 0.05059 | 1 | -3.76 | 0.00 |
| `claude-haiku-5-5` | medium | dlist3 | matched_medium_budget | wb_clean | 20 | 4.50/4.55 | 0.95/1.00 | 1273/133 | 0.887 | 1.000 | +0.113 [+0.050, +0.185] | 0.0009998 | 0.0009998 | 0.0009998 | +1.10 | 0.00 |
| `claude-haiku-5-5` | medium | dlist3 | matched_medium_budget | wb_decoy_trivial_guard | 20 | 4.50/4.75 | 0.95/1.00 | 1273/1156 | 0.887 | 0.840 | -0.046 [-0.177, +0.080] | 0.5125 | 0.5125 | 0.5125 | -1.27 | 0.30 |
| `claude-haiku-5-5` | medium | dlist3 | matched_medium_budget | wb_inv_code | 20 | 4.50/4.95 | 0.95/1.00 | 1273/1083 | 0.887 | 0.748 | -0.139 [-0.260, -0.018] | 0.0218 | 0.04359 | 0.7628 | -2.05 | 0.50 |
| `claude-haiku-5-5` | medium | dlist3 | matched_medium_budget | wb_inv_relu | 20 | 4.50/4.90 | 0.95/0.95 | 1273/934 | 0.887 | 0.900 | +0.014 [-0.076, +0.108] | 0.5767 | 0.5767 | 1 | -0.74 | 0.00 |
| `claude-haiku-5-5` | medium | majority3 | matched_medium_budget | wb_clean | 25 | 3.68/4.84 | 0.96/1.00 | 1409/155 | 0.972 | 1.000 | +0.028 [+0.000, +0.077] | 0.2416 | 0.2416 | 0.2416 | +0.85 | 0.00 |
| `claude-haiku-5-5` | medium | majority3 | matched_medium_budget | wb_decoy_trivial_guard | 25 | 3.68/5.00 | 0.96/1.00 | 1409/700 | 0.972 | 0.978 | +0.006 [-0.059, +0.068] | 1 | 1 | 1 | +0.22 | 0.04 |
| `claude-haiku-5-5` | medium | majority3 | matched_medium_budget | wb_inv_code | 25 | 3.68/4.88 | 0.96/1.00 | 1409/690 | 0.972 | 0.867 | -0.105 [-0.187, -0.034] | 0.0158 | 0.03159 | 0.5687 | -0.67 | 0.28 |
| `claude-haiku-5-5` | medium | majority3 | matched_medium_budget | wb_inv_relu | 25 | 3.68/5.00 | 0.96/1.00 | 1409/742 | 0.972 | 0.991 | +0.019 [-0.019, +0.070] | 0.7532 | 0.7532 | 1 | +0.55 | 0.00 |
| `claude-haiku-5-5` | medium | majority3 | matched_query_free_code | wb_clean | 25 | 3.24/4.80 | 0.80/1.00 | 1594/196 | 0.813 | 1.000 | +0.187 [+0.095, +0.291] | 0.003199 | 0.003199 | 0.003199 | +1.34 | 0.00 |
| `claude-haiku-5-5` | medium | majority3 | matched_query_free_code | wb_decoy_trivial_guard | 25 | 3.24/6.52 | 0.80/1.00 | 1594/681 | 0.813 | 1.000 | +0.187 [+0.095, +0.291] | 0.005799 | 0.005799 | 0.005799 | +1.34 | 0.00 |
| `claude-haiku-5-5` | medium | majority3 | matched_query_free_code | wb_inv_code | 25 | 3.24/6.32 | 0.80/1.00 | 1594/780 | 0.813 | 1.000 | +0.187 [+0.095, +0.291] | 1 | 1 | 1 | +1.34 | 0.00 |
| `claude-haiku-5-5` | medium | majority3 | matched_query_free_code | wb_inv_relu | 25 | 3.24/6.68 | 0.80/1.00 | 1594/691 | 0.813 | 1.000 | +0.187 [+0.095, +0.291] | 1 | 1 | 1 | +1.34 | 0.00 |
| `claude-haiku-5-5` | medium | majority3 | matched_total_budget | wb_clean | 25 | 2.80/3.00 | 0.80/1.00 | 1226/245 | 0.836 | 1.000 | +0.164 [+0.080, +0.259] | 0.002 | 0.002 | 0.002 | +1.03 | 0.00 |
| `claude-haiku-5-5` | medium | majority3 | matched_total_budget | wb_decoy_trivial_guard | 25 | 2.80/3.00 | 0.80/1.00 | 1226/541 | 0.836 | 0.880 | +0.044 [-0.077, +0.167] | 0.5053 | 0.5053 | 0.5053 | -0.38 | 0.24 |
| `claude-haiku-5-5` | medium | majority3 | matched_total_budget | wb_inv_code | 25 | 2.80/3.00 | 0.80/1.00 | 1226/498 | 0.836 | 0.566 | -0.270 [-0.371, -0.158] | 0.0002 | 0.0003999 | 0.009998 | -3.93 | 0.88 |
| `claude-haiku-5-5` | medium | majority3 | matched_total_budget | wb_inv_relu | 25 | 2.80/3.00 | 0.80/0.88 | 1226/560 | 0.836 | 0.562 | -0.274 [-0.386, -0.152] | 0.0003999 | 0.0003999 | 0.018 | -4.02 | 0.00 |
| `claude-opus-4-6` | medium | dlist3 | matched_medium_budget | wb_clean | 13 | 4.08/2.00 | 0.54/1.00 | 4208/0 | 0.576 | 1.000 | +0.424 [+0.330, +0.498] | 0.0003999 | 0.0003999 | 0.0003999 | +1.26 | 0.00 |
| `claude-opus-4-6` | medium | dlist3 | matched_medium_budget | wb_decoy_trivial_guard | 12 | 4.33/5.00 | 0.58/0.42 | 4217/3724 | 0.583 | 0.629 | +0.047 [-0.070, +0.177] | 0.4537 | 0.4537 | 0.4537 | -2.98 | 0.17 |
| `claude-opus-4-6` | medium | dlist3 | matched_medium_budget | wb_inv_code | 12 | 4.33/4.75 | 0.58/1.00 | 4217/555 | 0.583 | 0.848 | +0.266 [+0.148, +0.385] | 0.999 | 1 | 1 | -1.15 | 0.17 |
| `claude-opus-4-6` | medium | dlist3 | matched_medium_budget | wb_inv_relu | 12 | 4.33/5.00 | 0.58/0.92 | 4217/1312 | 0.583 | 0.958 | +0.376 [+0.273, +0.462] | 1 | 1 | 1 | -0.07 | 0.00 |
| `claude-opus-4-6` | medium | majority3 | matched_medium_budget | wb_clean | 25 | 5.00/2.00 | 1.00/1.00 | 244/0 | 0.928 | 1.000 | +0.072 [+0.008, +0.148] | 0.06239 | 0.06239 | 0.06239 | +0.58 | 0.00 |
| `claude-opus-4-6` | medium | majority3 | matched_medium_budget | wb_decoy_trivial_guard | 25 | 5.00/4.84 | 1.00/0.72 | 244/1698 | 0.928 | 0.795 | -0.133 [-0.257, -0.005] | 0.05139 | 0.05139 | 0.05139 | -1.50 | 0.12 |
| `claude-opus-4-6` | medium | majority3 | matched_medium_budget | wb_inv_code | 25 | 5.00/4.88 | 1.00/0.52 | 244/2520 | 0.928 | 0.642 | -0.286 [-0.381, -0.190] | 0.0002 | 0.0003999 | 0.009998 | -2.94 | 0.24 |
| `claude-opus-4-6` | medium | majority3 | matched_medium_budget | wb_inv_relu | 25 | 5.00/5.00 | 1.00/1.00 | 244/68 | 0.928 | 1.000 | +0.072 [+0.008, +0.148] | 1 | 1 | 1 | +0.58 | 0.00 |
| `claude-opus-4-8` | medium | majority3 | matched_medium_budget | wb_clean | 25 | 4.96/2.00 | 1.00/1.00 | 768/16 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | 1 | 1 | 1 | +0.58 | 0.00 |
| `claude-opus-4-8` | medium | majority3 | matched_medium_budget | wb_decoy_trivial_guard | 25 | 4.96/4.68 | 1.00/1.00 | 768/583 | 1.000 | 0.980 | -0.020 [-0.060, +0.000] | 1 | 1 | 1 | -0.03 | 0.00 |
| `claude-opus-4-8` | medium | majority3 | matched_medium_budget | wb_inv_code | 25 | 4.96/5.00 | 1.00/1.00 | 768/465 | 1.000 | 0.916 | -0.084 [-0.168, -0.018] | 0.06519 | 0.1304 | 1 | -0.55 | 0.16 |
| `claude-opus-4-8` | medium | majority3 | matched_medium_budget | wb_inv_relu | 25 | 4.96/3.64 | 1.00/1.00 | 768/397 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | 1 | 1 | 1 | +0.58 | 0.00 |
| `claude-opus-5-5` | medium | majority3 | matched_medium_budget | wb_clean | 25 | 4.16/4.60 | 1.00/1.00 | 223/60 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | 1 | 1 | 1 | +1.05 | 0.00 |
| `claude-opus-5-5` | medium | majority3 | matched_medium_budget | wb_decoy_trivial_guard | 25 | 4.16/4.60 | 1.00/1.00 | 223/205 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | 1 | 1 | 1 | +1.05 | 0.00 |
| `claude-opus-5-5` | medium | majority3 | matched_medium_budget | wb_inv_code | 25 | 4.16/5.00 | 1.00/1.00 | 223/165 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | 1 | 1 | 1 | +1.05 | 0.00 |
| `claude-opus-5-5` | medium | majority3 | matched_medium_budget | wb_inv_relu | 25 | 4.16/4.32 | 1.00/1.00 | 223/90 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | 1 | 1 | 1 | +1.05 | 0.00 |
| `claude-opus-5-5` | medium | majority3 | matched_total_budget | wb_clean | 25 | 3.00/3.00 | 1.00/1.00 | 186/72 | 0.533 | 1.000 | +0.467 [+0.414, +0.511] | 0.0002 | 0.0002 | 0.0002 | +1.20 | 0.00 |
| `claude-opus-5-5` | medium | majority3 | matched_total_budget | wb_decoy_trivial_guard | 25 | 3.00/3.00 | 1.00/1.00 | 186/158 | 0.533 | 0.963 | +0.430 [+0.331, +0.504] | 0.0002 | 0.0002 | 0.0002 | +0.44 | 0.08 |
| `claude-opus-5-5` | medium | majority3 | matched_total_budget | wb_inv_code | 25 | 3.00/3.00 | 1.00/1.00 | 186/122 | 0.533 | 1.000 | +0.467 [+0.414, +0.511] | 1 | 1 | 1 | +1.20 | 0.00 |
| `claude-opus-5-5` | medium | majority3 | matched_total_budget | wb_inv_relu | 25 | 3.00/3.00 | 1.00/1.00 | 186/53 | 0.533 | 0.947 | +0.414 [+0.339, +0.480] | 1 | 1 | 1 | +0.29 | 0.00 |
| `claude-sonnet-4-5-20250929` | default | majority3 | matched_medium_budget | wb_clean | 25 | 5.00/4.92 | 0.80/1.00 | 0/0 | 0.882 | 1.000 | +0.118 [+0.045, +0.206] | 0.0128 | 0.0128 | 0.0128 | +1.07 | 0.00 |
| `claude-sonnet-4-5-20250929` | default | majority3 | matched_medium_budget | wb_decoy_trivial_guard | 25 | 5.00/5.00 | 0.80/0.84 | 0/0 | 0.882 | 0.500 | -0.382 [-0.464, -0.288] | 0.0002 | 0.0002 | 0.0002 | -6.84 | 0.84 |
| `claude-sonnet-4-5-20250929` | default | majority3 | matched_medium_budget | wb_inv_code | 25 | 5.00/5.00 | 0.80/1.00 | 0/0 | 0.882 | 0.504 | -0.378 [-0.462, -0.282] | 0.0002 | 0.0003999 | 0.009998 | -6.84 | 1.00 |
| `claude-sonnet-4-5-20250929` | default | majority3 | matched_medium_budget | wb_inv_relu | 25 | 5.00/5.00 | 0.80/0.16 | 0/0 | 0.882 | 0.489 | -0.393 [-0.472, -0.303] | 0.0002 | 0.0003999 | 0.009998 | -6.84 | 0.00 |
| `claude-sonnet-4-6` | medium | majority3 | matched_medium_budget | wb_clean | 25 | 5.00/2.00 | 1.00/1.00 | 1573/71 | 0.689 | 1.000 | +0.311 [+0.213, +0.407] | 0.0002 | 0.0002 | 0.0002 | +0.67 | 0.00 |
| `claude-sonnet-4-6` | medium | majority3 | matched_medium_budget | wb_decoy_trivial_guard | 25 | 5.00/4.88 | 1.00/1.00 | 1573/2196 | 0.689 | 0.884 | +0.195 [+0.067, +0.321] | 0.006399 | 0.006399 | 0.006399 | -0.70 | 0.24 |
| `claude-sonnet-4-6` | medium | majority3 | matched_medium_budget | wb_inv_code | 25 | 5.00/5.00 | 1.00/1.00 | 1573/1446 | 0.689 | 0.943 | +0.254 [+0.124, +0.377] | 1 | 1 | 1 | -0.28 | 0.12 |
| `claude-sonnet-4-6` | medium | majority3 | matched_medium_budget | wb_inv_relu | 25 | 5.00/4.92 | 1.00/0.80 | 1573/2808 | 0.689 | 0.816 | +0.127 [-0.029, +0.278] | 0.9426 | 1 | 1 | -1.28 | 0.00 |
| `claude-sonnet-5-5` | high | majority3 | matched_medium_budget | wb_decoy_trivial_guard | 25 | 4.76/4.96 | 1.00/1.00 | 55/294 | 0.968 | 0.923 | -0.045 [-0.124, +0.025] | 0.2476 | 0.2476 | 0.2476 | -0.46 | 0.16 |
| `claude-sonnet-5-5` | high | majority3 | matched_medium_budget | wb_inv_code | 25 | 4.76/4.92 | 1.00/1.00 | 55/308 | 0.968 | 0.844 | -0.124 [-0.221, -0.030] | 0.006999 | 0.014 | 0.2659 | -1.07 | 0.32 |
| `claude-sonnet-5-5` | high | majority3 | matched_medium_budget | wb_inv_relu | 25 | 4.76/4.28 | 1.00/1.00 | 55/65 | 0.968 | 0.980 | +0.012 [-0.046, +0.066] | 0.6241 | 0.6241 | 1 | +0.01 | 0.00 |
| `claude-sonnet-5-5` | low | majority3 | matched_medium_budget | wb_decoy_trivial_guard | 25 | 4.80/4.88 | 1.00/1.00 | 0/121 | 1.000 | 0.858 | -0.142 [-0.239, -0.059] | 0.017 | 0.017 | 0.017 | -1.01 | 0.28 |
| `claude-sonnet-5-5` | low | majority3 | matched_medium_budget | wb_inv_code | 25 | 4.80/4.96 | 1.00/1.00 | 0/291 | 1.000 | 0.672 | -0.328 [-0.416, -0.235] | 0.0002 | 0.0003999 | 0.009998 | -2.54 | 0.68 |
| `claude-sonnet-5-5` | low | majority3 | matched_medium_budget | wb_inv_relu | 25 | 4.80/4.76 | 1.00/1.00 | 0/0 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | 1 | 1 | 1 | +0.58 | 0.00 |
| `claude-sonnet-5-5` | medium | dlist3 | matched_medium_budget | wb_clean | 20 | 4.50/2.45 | 1.00/0.95 | 217/2 | 0.979 | 0.975 | -0.004 [-0.075, +0.063] | 0.757 | 0.757 | 0.757 | -0.45 | 0.00 |
| `claude-sonnet-5-5` | medium | dlist3 | matched_medium_budget | wb_decoy_trivial_guard | 20 | 4.50/4.80 | 1.00/1.00 | 217/353 | 0.979 | 0.683 | -0.295 [-0.413, -0.171] | 0.0002 | 0.0002 | 0.0002 | -3.14 | 0.65 |
| `claude-sonnet-5-5` | medium | dlist3 | matched_medium_budget | wb_inv_code | 20 | 4.50/5.00 | 1.00/1.00 | 217/362 | 0.979 | 0.668 | -0.311 [-0.409, -0.210] | 0.0003999 | 0.0007998 | 0.018 | -3.24 | 0.70 |
| `claude-sonnet-5-5` | medium | dlist3 | matched_medium_budget | wb_inv_relu | 20 | 4.50/4.10 | 1.00/1.00 | 217/247 | 0.979 | 1.000 | +0.021 [+0.000, +0.063] | 1 | 1 | 1 | +0.58 | 0.00 |
| `claude-sonnet-5-5` | medium | majority3 | matched_medium_budget | wb_clean | 25 | 4.92/2.60 | 1.00/1.00 | 6/8 | 0.971 | 1.000 | +0.029 [+0.000, +0.086] | 0.4997 | 0.4997 | 0.4997 | +0.58 | 0.00 |
| `claude-sonnet-5-5` | medium | majority3 | matched_medium_budget | wb_decoy_trivial_guard | 25 | 4.92/4.92 | 1.00/1.00 | 6/127 | 0.971 | 0.964 | -0.007 [-0.055, +0.034] | 1 | 1 | 1 | -0.16 | 0.08 |
| `claude-sonnet-5-5` | medium | majority3 | matched_medium_budget | wb_inv_code | 25 | 4.92/5.00 | 1.00/1.00 | 6/258 | 0.971 | 0.761 | -0.210 [-0.317, -0.106] | 0.0003999 | 0.0007998 | 0.018 | -1.70 | 0.44 |
| `claude-sonnet-5-5` | medium | majority3 | matched_medium_budget | wb_inv_relu | 25 | 4.92/4.32 | 1.00/1.00 | 6/0 | 0.971 | 1.000 | +0.029 [+0.000, +0.086] | 1 | 1 | 1 | +0.58 | 0.00 |
| `claude-sonnet-5-5` | medium | majority3 | matched_query_free_code | wb_clean | 25 | 4.00/2.40 | 1.00/1.00 | 109/2 | 0.587 | 1.000 | +0.413 [+0.331, +0.484] | 0.0002 | 0.0002 | 0.0002 | +0.58 | 0.00 |
| `claude-sonnet-5-5` | medium | majority3 | matched_query_free_code | wb_decoy_trivial_guard | 25 | 4.00/5.20 | 1.00/1.00 | 109/122 | 0.587 | 1.000 | +0.413 [+0.331, +0.484] | 0.0002 | 0.0002 | 0.0002 | +0.58 | 0.00 |
| `claude-sonnet-5-5` | medium | majority3 | matched_query_free_code | wb_inv_code | 25 | 4.00/5.80 | 1.00/1.00 | 109/129 | 0.587 | 1.000 | +0.413 [+0.331, +0.484] | 1 | 1 | 1 | +0.58 | 0.00 |
| `claude-sonnet-5-5` | medium | majority3 | matched_query_free_code | wb_inv_relu | 25 | 4.00/4.16 | 1.00/1.00 | 109/2 | 0.587 | 1.000 | +0.413 [+0.331, +0.484] | 1 | 1 | 1 | +0.58 | 0.00 |
| `claude-sonnet-5-5` | medium | majority3 | matched_total_budget | wb_clean | 25 | 3.00/2.08 | 1.00/1.00 | 126/6 | 0.515 | 1.000 | +0.485 [+0.454, +0.516] | 0.0002 | 0.0002 | 0.0002 | +0.58 | 0.00 |
| `claude-sonnet-5-5` | medium | majority3 | matched_total_budget | wb_decoy_trivial_guard | 25 | 3.00/3.00 | 1.00/1.00 | 126/277 | 0.515 | 0.681 | +0.166 [+0.060, +0.274] | 0.006599 | 0.006599 | 0.006599 | -2.44 | 0.64 |
| `claude-sonnet-5-5` | medium | majority3 | matched_total_budget | wb_inv_code | 25 | 3.00/3.00 | 1.00/1.00 | 126/186 | 0.515 | 0.504 | -0.012 [-0.052, +0.032] | 0.3007 | 0.6015 | 1 | -7.32 | 1.00 |
| `claude-sonnet-5-5` | medium | majority3 | matched_total_budget | wb_inv_relu | 25 | 3.00/3.00 | 1.00/1.00 | 126/139 | 0.515 | 0.536 | +0.021 [-0.020, +0.071] | 0.773 | 0.773 | 1 | -5.22 | 0.00 |
