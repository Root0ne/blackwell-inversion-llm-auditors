# Pilot Analysis Report — Observer-Relative VBB & Computational Blackwell Inversion

- **Total API spend:** `$2.3895`
- **Exp 1 records:** `480` | **Leak Gate:** `PASS` | **H1 Verdict:** `REFUTED_VBB_BREACHED`
- **Exp 2 records:** `200` | **H2 Verdict:** `SUPPORTED_MECH1_ONLY`

## Experiment 1: Observer-Relative VBB (`C_kXOR`)

| Model | Track | Regime | Mode | Substrate | N | Adv (95% CI) | Delta_WB | p_holm |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| `claude-haiku-5-5` | tool_python | control_impossible | BB | blackbox | 30 | +0.000 [-0.204, +0.204] | +0.000 | 1 |
| `claude-haiku-5-5` | tool_python | control_impossible | WB | code_clean | 30 | +0.000 [-0.204, +0.204] | +0.000 | 1 |
| `claude-haiku-5-5` | tool_python | control_impossible | WB | code_obf | 30 | +0.000 [-0.204, +0.204] | +0.000 | 1 |
| `claude-haiku-5-5` | tool_python | control_impossible | WB | relu_weights | 30 | +0.133 [-0.190, +0.423] | +0.133 | 1 |
| `claude-haiku-5-5` | tool_python | transition | BB | blackbox | 30 | +0.000 [-0.204, +0.204] | +0.000 | - |
| `claude-haiku-5-5` | tool_python | transition | WB | code_clean | 30 | +0.800 [+0.463, +0.911] | +0.800 | 7.654e-05 |
| `claude-haiku-5-5` | tool_python | transition | WB | code_obf | 30 | +0.667 [+0.324, +0.834] | +0.667 | 0.000726 |
| `claude-haiku-5-5` | tool_python | transition | WB | relu_weights | 30 | +0.467 [+0.112, +0.690] | +0.467 | 0.02533 |
| `claude-haiku-5-5` | tool_python | transparent | BB | blackbox | 30 | +0.000 [-0.204, +0.204] | +0.000 | - |
| `claude-haiku-5-5` | tool_python | transparent | WB | code_clean | 30 | +1.000 [+0.712, +1.000] | +1.000 | 5.802e-08 |
| `claude-haiku-5-5` | tool_python | transparent | WB | code_obf | 30 | +0.733 [+0.386, +0.869] | +0.733 | 0.0003472 |
| `claude-haiku-5-5` | tool_python | transparent | WB | relu_weights | 30 | +0.733 [+0.391, +0.874] | +0.733 | 0.0003472 |
| `claude-haiku-5-5` | tool_python | vbb_hard | BB | blackbox | 30 | +0.000 [-0.204, +0.204] | +0.000 | - |
| `claude-haiku-5-5` | tool_python | vbb_hard | WB | code_clean | 30 | +1.000 [+0.712, +1.000] | +1.000 | 5.802e-08 |
| `claude-haiku-5-5` | tool_python | vbb_hard | WB | code_obf | 30 | +0.733 [+0.391, +0.874] | +0.733 | 0.0003472 |
| `claude-haiku-5-5` | tool_python | vbb_hard | WB | relu_weights | 30 | +0.267 [-0.081, +0.538] | +0.267 | 0.1362 |

## Experiment 2: Computational Blackwell Inversion (`C_inv`)

| Model | Budget Track | Variant | Pairs | BB Acc | WB Acc | Delta_WB (95% CI) | p_holm | Gamma_WB (bits) | Decoy Trap |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `claude-haiku-5-5` | matched_query_free_code | wb_clean | 25 | 0.900 | 1.000 | +0.100 [+0.020, +0.180] | 0.02939 | +1.08 | 0.00 |
| `claude-haiku-5-5` | matched_query_free_code | wb_inv_code | 25 | 0.900 | 0.980 | +0.080 [+0.000, +0.180] | 1 | +0.41 | 0.00 |
| `claude-haiku-5-5` | matched_query_free_code | wb_inv_relu | 25 | 0.900 | 0.971 | +0.071 [+0.002, +0.151] | 1 | +0.29 | 0.00 |
| `claude-haiku-5-5` | matched_total_budget | wb_clean | 25 | 0.741 | 1.000 | +0.259 [+0.162, +0.357] | 0.0002 | +1.37 | 0.00 |
| `claude-haiku-5-5` | matched_total_budget | wb_inv_code | 25 | 0.741 | 0.542 | -0.199 [-0.321, -0.071] | 0.0112 | -4.16 | 0.88 |
| `claude-haiku-5-5` | matched_total_budget | wb_inv_relu | 25 | 0.741 | 0.648 | -0.093 [-0.208, +0.027] | 0.2118 | -2.04 | 0.00 |
