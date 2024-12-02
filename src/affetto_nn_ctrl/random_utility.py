from __future__ import annotations

import numpy as np
from numpy.random import Generator, default_rng

from affetto_nn_ctrl._typing import NoDefault, no_default

__SEED: int | None = None
__GLOBAL_RNG: Generator = default_rng()


def set_seed(seed: int | None) -> None:
    global __SEED  # noqa: PLW0603
    global __GLOBAL_RNG  # noqa: PLW0603
    if isinstance(seed, int | None):
        __SEED = seed
        __GLOBAL_RNG = default_rng(__SEED)
        np.random.seed(__SEED)  # noqa: NPY002
    else:
        msg = f"`seed` is expected to be an `int` or `None`, not {type(seed)}"
        raise TypeError(msg)


def get_seed() -> int | None:
    return __SEED


def get_rng(seed: int | Generator | NoDefault | None = no_default) -> Generator:
    if seed is no_default:
        return __GLOBAL_RNG
    if isinstance(seed, Generator):
        return seed
    if isinstance(seed, int | None):
        return default_rng(seed)
    msg = f"`seed` is expected to be an `int` or `None`, not {type(seed)}"
    raise TypeError(msg)


# Local Variables:
# jinx-local-words: "noqa"
# End:
