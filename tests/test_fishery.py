import random

from nem.env.fishery import allocate_harvests, regrow, step_population


def test_regrow_logistic_below_capacity():
    # 1500 + 0.3 * 1500 * (1 - 0.5) = 1725
    assert regrow(1500) == 1725


def test_regrow_at_capacity_stays_at_capacity():
    assert regrow(3000) == 3000


def test_regrow_zero_stays_zero():
    assert regrow(0) == 0


def test_allocation_fills_requests_when_stock_suffices():
    got = allocate_harvests({0: 50, 1: 30}, population=1000, rng=random.Random(0))
    assert got == {0: 50, 1: 30}


def test_allocation_clamps_requests_to_max_harvest_and_nonnegative():
    got = allocate_harvests({0: 500, 1: -5}, population=1000, rng=random.Random(0))
    assert got == {0: 100, 1: 0}


def test_allocation_random_order_never_exceeds_population():
    got = allocate_harvests({i: 100 for i in range(4)}, population=250, rng=random.Random(1))
    assert sum(got.values()) == 250
    assert sorted(got.values()) == [0, 50, 100, 100]


def test_step_population_collapse_below_threshold():
    nxt, collapsed = step_population(population=150, total_harvest=60)
    assert collapsed is True
    assert nxt == 0


def test_step_population_regrows_after_harvest():
    nxt, collapsed = step_population(population=3000, total_harvest=1500)
    assert collapsed is False
    assert nxt == 1725
