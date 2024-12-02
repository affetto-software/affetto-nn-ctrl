from __future__ import annotations

from affetto_nn_ctrl.random_utility import get_rng, set_seed


def test_set_seed() -> None:
    set_seed(0)
    rng = get_rng()
    n = 10
    numbers1 = [rng.uniform(0.0, 1.0) for _ in range(n)]
    set_seed(0)
    rng = get_rng()
    numbers2 = [rng.uniform(0.0, 1.0) for _ in range(n)]
    assert numbers1 == numbers2


def test_get_rng() -> None:
    set_seed(0)
    rng1 = get_rng()
    rng2 = get_rng()
    assert rng1 is rng2


def test_get_rng_reset_seed() -> None:
    set_seed(0)
    rng1 = get_rng()
    set_seed(0)
    rng2 = get_rng()
    assert rng1 is not rng2
