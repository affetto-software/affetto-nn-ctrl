# ruff: noqa: N803, N802, N806, NPY002, ANN204, ANN205, ANN201, C901, PLR0912, ERA001
from __future__ import annotations

from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse, stats
from scipy.sparse import linalg as sla
from sklearn.metrics import r2_score

from affetto_nn_ctrl.random_utility import get_rng

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyplotutil.datautil import Unknown


def noise(
    dist: str = "normal",
    size: int | tuple[int] = 1,
    gain: float = 1.0,
    seed: int | None = None,
    **kwargs: Unknown,
) -> np.ndarray:
    if abs(gain) > 0.0:
        rng = get_rng(seed)
        return gain * getattr(rng, dist)(**kwargs, size=size)
    return np.zeros(shape=size)


class Layer:
    weight: np.ndarray
    keys: list[str] | None

    def __init__(
        self,
        N_in: int,
        N_out: int,
        scale: float = 1.0,
        dist: str = "uniform",
        keys: list[str] | None = None,
    ) -> None:
        size = (N_out, N_in)
        if dist == "normal":
            self.weight = scale * np.random.normal(size=size)
        else:
            self.weight = np.random.uniform(low=-scale, high=scale, size=size)
        self.set_keys(keys)

    def __call__(self, _input: np.ndarray):
        return np.dot(self.weight, _input)

    def set_weight(self, weight: np.ndarray) -> None:
        self.weight = weight

    def set_keys(self, keys: list[str] | None) -> None:
        if keys is not None:
            self.keys = keys.copy()
        else:
            self.keys = None


class Reservoir:
    N_x: int
    density: float
    rho: float
    activation_func: Callable
    alpha: float
    W: sparse.csr_matrix
    x: np.ndarray
    internal_state: np.ndarray

    def __init__(
        self,
        N_x: int,
        density: float,
        rho: float,
        leaking_rate: float,
        activation_func: Callable,
        *,
        randomize_initial_state: bool = False,
    ) -> None:
        self.N_x = N_x
        self.density = density
        self.rho = rho
        self.alpha = leaking_rate
        self.activation_func = activation_func
        self.resample_connection()
        self.reset_reservoir_state(randomize_initial_state=randomize_initial_state)
        self.internal_state = np.zeros((N_x,), dtype=float)
        self.forward = self.forward_internal

    @staticmethod
    def _getrvs(dist: str, **kwargs: Unknown):
        distribution = getattr(stats, dist)
        return partial(distribution(**kwargs).rvs)

    @staticmethod
    def spectral_radius(W: sparse.csr_matrix | sparse.coo_matrix | sparse.spmatrix):
        eigvals = sla.eigs(W, k=1, which="LM", return_eigenvectors=False)
        return np.max(np.abs(eigvals))

    def make_connection(self, N_x: int, density: float, rho: float, scale: float = 1.0):
        scale = abs(scale)
        rvs = self._getrvs("uniform", loc=-scale, scale=2.0 * scale)
        W = sparse.random(N_x, N_x, density=density, format="csr", data_rvs=rvs, dtype=float)
        current_sr = self.spectral_radius(W)
        return W.multiply(rho / current_sr)

    def resample_connection(self):
        self.W = self.make_connection(self.N_x, self.density, self.rho)
        return self.W

    def reset_reservoir_state(self, *, randomize_initial_state: bool = False, scale: float = 1.0):
        if randomize_initial_state:
            self.x = np.random.uniform(-scale, scale, (self.N_x,))
        else:
            self.x = np.zeros((self.N_x,), dtype=float)
        return self.x

    def kernel(self, x_in: np.ndarray, x: np.ndarray):
        return x_in + self.W.dot(x)

    def forward_internal(self, x_in: np.ndarray):
        lr = self.alpha
        f = self.activation_func
        x = self.x

        s_next = (1.0 - lr) * x + lr * f(self.kernel(x_in, x))
        self.x = s_next
        return self.x

    def forward_external(self, x_in: np.ndarray):
        lr = self.alpha
        f = self.activation_func
        x = self.x
        s = self.internal_state

        s_next = (1.0 - lr) * s + lr * self.kernel(x_in, x)
        self.internal_state = s_next
        self.x = f(s_next)
        return self.x

    def __call__(self, x_in: np.ndarray):
        return self.forward(x_in)


class Optimizer:
    @abstractmethod
    def __call__(self, d: np.ndarray, x: np.ndarray) -> Unknown: ...

    @abstractmethod
    def get_Wout(self) -> np.ndarray: ...


class Tikhonov(Optimizer):
    def __init__(self, N_x: int, N_y: int, beta: float) -> None:
        self.N_x = N_x
        self.N_y = N_y
        self.beta = beta
        self.X_XT = np.zeros((N_x, N_x), dtype=float)
        self.D_XT = np.zeros((N_y, N_x), dtype=float)

    def __call__(self, d: np.ndarray, x: np.ndarray) -> None:
        x = np.reshape(x, (-1, 1))
        d = np.reshape(d, (-1, 1))
        self.X_XT += np.dot(x, x.T)
        self.D_XT += np.dot(d, x.T)

    def get_Wout(self) -> np.ndarray:
        X_pseudo_inv = np.linalg.inv(self.X_XT + self.beta * np.identity(self.N_x, dtype=float))
        return np.dot(self.D_XT, X_pseudo_inv)


class RLS(Optimizer):
    def __init__(self, N_x: int, N_y: int, delta: float, lam: float, update: int) -> None:
        self.delta = delta
        self.lam = lam
        self.lam_inv = 1.0 / lam
        self.update = update
        self.P = (1.0 / self.delta) * np.identity(N_x, dtype=float)
        self.Wout = np.zeros((N_y, N_x), dtype=float)

    def __call__(self, d: np.ndarray, x: np.ndarray) -> np.ndarray:
        x = np.reshape(x, (-1, 1))
        d = np.reshape(d, (-1, 1))
        for _ in np.arange(self.update):
            v = d - np.dot(self.Wout, x)
            gain = self.lam_inv * np.dot(self.P, x)
            gain = gain / (1.0 + self.lam_inv * np.dot(np.dot(x.T, self.P), x))
            self.P = self.lam_inv * (self.P - np.dot(np.dot(gain, x.T), self.P))
            self.Wout += np.dot(v, gain.T)

        return self.Wout

    def get_Wout(self) -> np.ndarray:
        return self.Wout


class ESN:
    N_u: int
    N_y: int
    N_x: int
    input: Layer
    reservoir: Reservoir
    readout: Layer
    feedback: Layer | None
    optimizer: Optimizer | None
    input_noise: Callable
    feedback_noise: Callable
    u_prev: np.ndarray | None

    def __init__(
        self,
        N_u: int,
        N_y: int,
        N_x: int,
        *,
        density: float = 0.05,
        rho: float = 0.95,
        leaking_rate: float = 1.0,
        input_scale: float = 1.0,
        activation_func: Callable = np.tanh,
        fb_scale: float | None = None,
        randomize_initial_state: bool = False,
        noise_in: float = 0.0,
        noise_fb: float = 0.0,
        input_keys: list[str] | None = None,
        readout_keys: list[str] | None = None,
        optimizer: str | None = "Tikhonov",
        noise_in_seed: int | None = None,
        noise_fb_seed: int | None = None,
    ) -> None:
        self.N_u = N_u
        self.N_y = N_y
        self.N_x = N_x
        self.input = Layer(N_u, N_x, input_scale, keys=input_keys)
        self.reservoir = Reservoir(
            N_x,
            density,
            rho,
            leaking_rate,
            activation_func,
            randomize_initial_state=randomize_initial_state,
        )
        self.readout = Layer(N_x, N_y, dist="normal", keys=readout_keys)
        if fb_scale is None:
            self.feedback = None
        else:
            self.feedback = Layer(N_y, N_x, fb_scale, keys=readout_keys)
        self.input_noise = partial(noise, size=N_x, gain=noise_in, seed=noise_in_seed)
        self.feedback_noise = partial(noise, size=N_x, gain=noise_fb, seed=noise_fb_seed)
        if optimizer is not None:
            self.optimizer = self.select_optimizer(optimizer, N_x, N_y)
        else:
            self.optimizer = None
        self.u_prev = None

    def select_optimizer(self, optimizer: str, N_x: int, N_y: int) -> Optimizer:
        if "tikhonov".startswith(optimizer.lower()):
            self.optimizer = Tikhonov(N_x, N_y, 1e-4)
        elif "rls".startswith(optimizer.lower()):
            self.optimizer = RLS(N_x, N_y, 1e-4, 1.0, 1)
        else:
            msg = f"Unrecognized optimizer name: {optimizer}"
            raise NotImplementedError(msg)
        return self.optimizer

    def reset_reservoir_state(self, *, randomize_initial_state: bool = False) -> None:
        self.reservoir.reset_reservoir_state(randomize_initial_state=randomize_initial_state)

    @property
    def yinit(self) -> np.ndarray:
        return np.zeros((self.N_y,), dtype=float)

    @property
    def input_keys(self) -> list[str] | None:
        return self.input.keys

    @property
    def readout_keys(self) -> list[str] | None:
        return self.readout.keys

    def _fit(
        self,
        U: np.ndarray,
        D: np.ndarray,
        F: np.ndarray | None = None,
        y0: np.ndarray | None = None,
        optimizer: Optimizer | None = None,
        *,
        enable_teacher_force: bool = False,
        warmup: int = 0,
    ) -> np.ndarray:
        # initialize
        if optimizer is None:
            if self.optimizer is None:
                msg = "No optimizer is defined"
                raise RuntimeError(msg)
            optimizer = self.optimizer

        Y = []
        if y0 is not None:
            y = y0
        elif F is not None and len(F) > 0:
            y = F[0]
        else:
            y = self.yinit

        # check length
        assert len(U) == len(D)
        if F is not None:
            assert len(F) == len(D) + 1

        # warming up
        for i in range(warmup):  # noqa: B007
            # x_in = self.input(U[0])
            x_in = self.input(np.zeros((U.shape[1],)))
            # x_in = self.input(np.random.uniform((U.shape[1],)))
            x = self.reservoir(x_in)
            # Wout = optimizer(d, x)
            # y = self.readout(x)

        # train model
        for i, (u, d) in enumerate(zip(U, D, strict=False)):
            x_in = self.input(u) + self.input_noise()

            if self.feedback is not None:
                x_in += self.feedback(y) + self.feedback_noise()

            x = self.reservoir(x_in)
            Wout = optimizer(d, x)
            if Wout is None:
                y = self.readout(x)  # batch learning (Tikhonov)
                # Note: this output is meaningless in batch learning.
            else:
                y = np.dot(Wout, x)  # online learning (RLS)
            Y.append(y)

            # teacher forcing
            if enable_teacher_force:
                y = d
            elif F is not None:
                y = F[i + 1]

        self.readout.set_weight(optimizer.get_Wout())
        return np.array(Y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> ESN:
        self._fit(X, y, enable_teacher_force=True)
        return self

    def _predict(self, U: np.ndarray) -> np.ndarray:
        Y = []
        y = self.yinit
        for u in U:
            x_in = self.input(u)
            if self.feedback is not None:
                x_in += self.feedback(y)
            x = self.reservoir(x_in)
            y = self.readout(x)
            Y.append(y)
        return np.array(Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.reset_reservoir_state()
        return self._predict(X)

    def run(self, U: np.ndarray) -> np.ndarray:
        Y = []
        y = self.yinit
        u = U[0]
        assert u.shape == y.shape
        self.reset_reservoir_state()
        for _ in range(len(U)):
            x_in = self.input(u)
            if self.feedback is not None:
                x_in += self.feedback(y)
            x = self.reservoir(x_in)
            y = self.readout(x)
            Y.append(y)
            u = y
        return np.array(Y)

    def oneshot(
        self,
        u: np.ndarray,
        feedback: np.ndarray | None = None,
        *,
        reset_when_context_switch: bool = False,
    ) -> np.ndarray:
        if reset_when_context_switch and (self.u_prev is None or not np.allclose(u, self.u_prev)):
            self.reset_reservoir_state()
            self.u_prev = u.copy()

        x_in = self.input(u)
        if self.feedback is not None and feedback is not None:
            x_in += self.feedback(feedback)
        x = self.reservoir(x_in)
        return self.readout(x)

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ):
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)

    def get_params(self) -> dict[str, object]:
        return {}


# Local Variables:
# jinx-local-words: "Wout csr noqa np rls tikhonov"
# End:
