"""Per-model token prices used for the local spend cap and the `cost_usd` record field.

The rates are approximate list prices at the time of the runs; they drive the
--max-cost-usd guard, not billing.
"""

from __future__ import annotations

from typing import List, Tuple

# (model id prefix, USD per 1M input tokens, USD per 1M output tokens).
# The first matching prefix wins, so longer prefixes come before shorter ones.
MODEL_PRICING_USD_PER_MTOK: List[Tuple[str, float, float]] = [
    ("claude-haiku-5-5", 0.10, 0.50),
    ("claude-sonnet-5-5", 2.00, 10.00),
    ("claude-sonnet-5", 3.00, 15.00),
    ("claude-opus-5-5", 4.00, 20.00),
    ("claude-opus-5", 5.00, 25.00),
    ("claude-fable-5-2", 8.00, 40.00),
    ("claude-fable-5-1", 10.00, 50.00),
    ("claude-fable-5", 10.00, 50.00),
    ("claude-opus-4-8", 5.00, 25.00),
    ("claude-opus-4-7", 5.00, 25.00),
    ("claude-opus-4-6", 5.00, 25.00),
    ("claude-sonnet-4-6", 3.00, 15.00),
    ("claude-opus-4-5", 5.00, 25.00),
    ("claude-sonnet-4-5", 3.00, 15.00),
    ("claude-haiku-4-5", 1.00, 5.00),
    ("claude-opus-4", 15.00, 75.00),
    ("claude-sonnet-4", 3.00, 15.00),
    ("claude-3-7-sonnet", 3.00, 15.00),
    ("claude-3-5-sonnet", 3.00, 15.00),
    ("claude-3-5-haiku", 0.80, 4.00),
    # Google Gemini and open-weight Gemma (Nov 2026 rates; introductory rates active
    # through Nov 30 / Dec 31, 2026).
    ("gemini-4-carbon", 2.00, 10.00),
    ("gemini-4-argon", 2.00, 10.00),
    ("gemini-4-neon", 3.00, 15.00),
    ("gemini-4-hydrogen", 1.50, 7.50),
    ("gemini-3.8-flash", 0.75, 3.75),
    ("gemini-3.7-flash", 0.75, 3.75),
    ("gemini-3.6-flash", 0.75, 3.75),
    ("gemini-3.5-flash-lite", 0.30, 2.50),
    ("gemini-3.5-flash", 0.75, 3.75),
    ("gemini-3.1-flash-lite", 0.25, 1.50),
    ("gemini-2.5-flash-lite", 0.10, 0.40),
    ("gemini-2.5-flash", 0.30, 2.50),
    ("gemini-2.5-pro", 1.25, 10.00),
    ("gemma-4-", 0.0, 0.0),
    ("mock-", 0.0, 0.0),
]

# Deliberately high rates for model ids missing from the table, so the cap errs on the safe side.
FALLBACK_RATES_USD_PER_MTOK = (5.0, 25.0)
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """USD cost of one call; cache writes bill at 1.25x and cache reads at 0.10x the input rate."""
    m_low = model.lower()
    in_rate, out_rate = FALLBACK_RATES_USD_PER_MTOK
    for prefix, r_in, r_out in MODEL_PRICING_USD_PER_MTOK:
        if m_low.startswith(prefix):
            in_rate, out_rate = r_in, r_out
            break
    total_in_cost = (
        float(input_tokens) * in_rate
        + float(cache_write_tokens) * (CACHE_WRITE_MULTIPLIER * in_rate)
        + float(cache_read_tokens) * (CACHE_READ_MULTIPLIER * in_rate)
    )
    return (total_in_cost + float(output_tokens) * out_rate) / 1_000_000.0
