"""Model prices, used to report the cost of each configuration.

Prices are US dollars per million tokens (input, output), standard tier, as
listed on OpenAI's pricing and model pages when checked. They change, so
update them before relying on the cost columns; a model missing here is
reported without a cost rather than with a guess.
"""

PRICES_CHECKED_ON = "2026-11-04"

PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-terra": (1.00, 5.00),
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6-astra": (10.00, 50.00),
    "text-embedding-3-small": (0.02, 0.0),
}


def cost(model: str | None, input_tokens: int, output_tokens: int) -> float | None:
    if model not in PRICES_PER_MILLION:
        return None
    input_price, output_price = PRICES_PER_MILLION[model]
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
