"""Pure fishery dynamics (paper App. A.5). No LLM calls, no state mutation."""

import random

from nem.config import CARRYING_CAPACITY, COLLAPSE_THRESHOLD, MAX_HARVEST, REGEN_RATE


def regrow(population: float) -> float:
    """Logistic regrowth, capped at carrying capacity."""
    k = CARRYING_CAPACITY
    grown = population + REGEN_RATE * population * (1 - population / k)
    return min(k, grown)


def clamp_request(request: int) -> int:
    return max(0, min(MAX_HARVEST, int(request)))


def allocate_harvests(
    requests: dict[int, int], population: float, rng: random.Random
) -> dict[int, int]:
    """Serve clamped requests in uniformly random order until the stock runs out."""
    order = list(requests)
    rng.shuffle(order)
    remaining = int(population)
    allocated: dict[int, int] = {}
    for agent in order:
        take = min(clamp_request(requests[agent]), remaining)
        allocated[agent] = take
        remaining -= take
    return {agent: allocated[agent] for agent in requests}


def step_population(population: float, total_harvest: int) -> tuple[float, bool]:
    """Subtract harvest, then regrow. Returns (next_population, collapsed)."""
    after = max(0.0, population - total_harvest)
    if after < COLLAPSE_THRESHOLD:
        return 0, True
    return regrow(after), False
