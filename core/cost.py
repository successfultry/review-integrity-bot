from __future__ import annotations

from models.review import Usage


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int, input_price_per_1m: float, output_price_per_1m: float) -> float:
    in_cost = (prompt_tokens / 1_000_000) * input_price_per_1m
    out_cost = (completion_tokens / 1_000_000) * output_price_per_1m
    return round(in_cost + out_cost, 8)


def merge_usage(parts: list[Usage]) -> Usage:
    prompt_tokens = sum(p.prompt_tokens for p in parts)
    completion_tokens = sum(p.completion_tokens for p in parts)
    total_tokens = sum(p.total_tokens for p in parts)
    estimated_cost_usd = round(sum(p.estimated_cost_usd for p in parts), 8)
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd,
    )
