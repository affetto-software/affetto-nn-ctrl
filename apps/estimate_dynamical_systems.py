#!/usr/bin/env python
# ruff: noqa: N803, N806, ERA001
from __future__ import annotations

from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from affetto_nn_ctrl.esn import ESN
from affetto_nn_ctrl.random_utility import get_rng, set_seed

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from pyplotutil.datautil import Unknown


def lorenz(
    _: float,
    xyz: tuple[float, float, float],
    sigma: float,
    beta: float,
    rho: float,
) -> tuple[float, float, float]:
    x, y, z = xyz
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return (dx, dy, dz)


def vdp(_: float, uv: tuple[float, float], mu: float) -> tuple[float, float]:
    u, v = uv
    du = v
    dv = mu * (1.0 - u * u) * v - u
    return (du, dv)


def lq(_: float, state: tuple[float, float], a: float, b: float, c: float) -> tuple[float, float]:
    y, dy = state
    ddy = -(b * dy + c * y) / a
    return (dy, ddy)


class DynSys:
    def __init__(
        self,
        f: Callable,
        *params: float,
        y0_low: float | None = None,
        y0_high: float | None = None,
    ) -> None:
        self.f = f
        self.params = params
        self.dim = self.estimate_dim(f)
        self.y0_low = y0_low
        self.y0_high = y0_high
        self.rng = get_rng(12345)  # Get new RNG instance to generate consistent initial values.

    @staticmethod
    def estimate_dim(f: Callable) -> int:
        n_params = len(signature(f).parameters) - 2
        dummy_params = [1.0 for _ in range(n_params)]

        dim = 1
        found = False
        while not found:
            y = np.zeros((dim,))
            try:
                f(0.0, y, *dummy_params)
            except ValueError:
                dim += 1
            else:
                found = True
        return dim

    def _y0_lim(self, y0_low: float | None, y0_high: float | None) -> tuple[float, float]:
        if y0_low is None:
            y0_low = self.y0_low if self.y0_low is not None else 0.0
        if y0_high is None:
            y0_high = self.y0_high if self.y0_high is not None else 1.0
        return y0_low, y0_high

    def solve(
        self,
        T: float,
        dt: float = 1.0,
        y0: Iterable | None = None,
        y0_low: float | None = None,
        y0_high: float | None = None,
    ) -> Unknown:
        if y0 is None:
            y0_low, y0_high = self._y0_lim(y0_low, y0_high)
            y0 = self.rng.uniform(y0_low, y0_high, (self.dim,))
        else:
            y0 = np.array(y0, dtype=float)
        t_eval = np.arange(0.0, T, dt)
        solver = solve_ivp(self.f, (0.0, T), y0, t_eval=t_eval, args=self.params)
        return solver.y.T

    def __call__(
        self,
        T: float,
        dt: float = 1.0,
        y0: Iterable | None = None,
        y0_low: float | None = None,
        y0_high: float | None = None,
    ) -> Unknown:
        return self.solve(T, dt, y0, y0_low, y0_high)


def generate_random_initial_points(
    N: int,
    dim: int,
    low: Iterable[float] | None = None,
    high: Iterable[float] | None = None,
) -> list[np.ndarray]:
    size = (N, dim)
    kwargs = {}
    if low is not None:
        kwargs["low"] = low
    if high is not None:
        kwargs["high"] = high
    return list(get_rng().uniform(size=size, **kwargs))  # type: ignore[call-overload]


def _generate_trajectory(ds: DynSys, p0: Iterable[float], T: float, dt: float) -> np.ndarray:
    return ds(T + dt, dt, p0)


def generate_trajectories(
    ds: DynSys,
    p0_list: Iterable[Iterable[float]],
    T: float,
    dt: float = 1.0,
) -> list[np.ndarray]:
    return [_generate_trajectory(ds, p0, T, dt) for p0 in p0_list]


def prepare_data(
    ds: DynSys,
    n_train: int,
    n_test: int,
    T: float,
    dt: float = 1.0,
    xlim: tuple[float, float] = (-1.0, 1.0),
    ylim: tuple[float, float] = (-1.0, 1.0),
    zlim: tuple[float, float] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    dim = ds.dim
    if dim == 2:  # noqa: PLR2004
        low = (xlim[0], ylim[0])
        high = (xlim[1], ylim[1])
    elif dim == 3:  # noqa: PLR2004
        if zlim is None:
            msg = "zlim must be given as the dynamical systems is 3 dimensional."
            raise ValueError(msg)
        low = (xlim[0], ylim[0], zlim[0])  # type: ignore[assignment]
        high = (xlim[1], ylim[1], zlim[1])  # type: ignore[assignment]
    else:
        msg = f"Unsupported dimensions: {dim}"
        raise ValueError(msg)

    train_data_p0 = generate_random_initial_points(n_train, dim, low, high)
    train_data = generate_trajectories(ds, train_data_p0, T, dt)
    test_data_p0 = generate_random_initial_points(n_test, dim, low, high)
    test_data = generate_trajectories(ds, test_data_p0, T, dt)
    return train_data, test_data


def create_model(
    N_u: int,
    N_y: int,
    N_x: int,
    density: float,
    rho: float,
    leaking_rate: float,
    input_scale: float,
    noise_in: float,
    optimizer: str = "tikhonov",
) -> ESN:
    return ESN(
        N_u,
        N_y,
        N_x,
        density=density,
        rho=rho,
        leaking_rate=leaking_rate,
        input_scale=input_scale,
        noise_in=noise_in,
        optimizer=optimizer,
    )


def train(model: ESN, train_data: list[np.ndarray]) -> np.ndarray:
    y = np.empty(train_data[0].shape)
    for data in train_data:
        U, D = data[:-1], data[1:]
        model.reset_reservoir_state()
        y = model.fit(U, D, enable_teacher_force=True)
    return y


def mean_squared_error(
    y: Iterable[float | Iterable[float]],
    y_target: Iterable[float | Iterable[float]],
    *,
    root: bool = False,
) -> float:
    y = np.asarray(y, dtype=float)
    y_target = np.asarray(y_target, dtype=float)
    errors = np.average((y - y_target) ** 2, axis=0)
    if root:
        errors = np.sqrt(errors)
    return np.average(errors)


def plot2d(
    data: Iterable[np.ndarray],
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    filename: str | None = None,
    *,
    color: bool = False,
    ext: str | list[str] = ".png",
) -> tuple[Figure, Axes]:
    plt.rcParams["figure.dpi"] = 100
    fig = plt.figure(figsize=(7, 7))
    ax: Axes = fig.add_subplot(111)
    for xy in data:
        if color:
            (line,) = ax.plot(xy[:, 0][0], xy[:, 1][0], marker="x", ms=None)
        else:
            (line,) = ax.plot(xy[:, 0][0], xy[:, 1][0], marker="x", ms=None, c="k")
        ax.plot(xy[:, 0], xy[:, 1], lw=1, c=line.get_color())
    plt.xlabel("x")
    plt.ylabel("y")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title is not None:
        plt.title(title)
    ax.set_aspect("equal", "box")
    if filename is not None:
        if color:
            path = Path(filename)
            fixed_path = path.with_name(path.stem + "_color" + path.suffix)
            filename = str(fixed_path)
        if isinstance(ext, str):
            ext = [ext]
        for e in ext:
            plt.savefig(Path(filename).with_suffix(e), bbox_inches="tight")
    return fig, ax


def save_errors(errors: list[float], label: str, suffix: str = "") -> None:
    err = np.asarray(errors, dtype=float)
    err_avg = np.average(err)
    err_std = np.std(err)
    err_var = np.var(err)
    if suffix != "":
        suffix = f"_{suffix}"
    np.savetxt(f"{label}_error{suffix}.txt", err)
    np.savetxt(f"{label}_error{suffix}_stat.txt", [err_avg, err_std, err_var])


def estimate_ds_2d(
    f: Callable,
    f_params: tuple[float, ...],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    T: float,
    n_train: int,
    n_test: int,
    dt: float = 0.01,
    N_x: int = 300,
    density: float = 0.1,
    rho: float = 0.95,
    lr: float = 0.99,
    input_scale: float = 0.1,
    noise_in: float = 0.0,
    label: str | None = None,
    plot_xlim: tuple[float, float] | None = None,
    plot_ylim: tuple[float, float] | None = None,
    *,
    color: bool = False,
    ext: str | list[str] = ".png",
) -> None:
    # prepare data and model
    ds = DynSys(f, *f_params)
    train_data, test_data = prepare_data(ds, n_train, n_test, T, dt=dt, xlim=xlim, ylim=ylim)
    dim = train_data[0].shape[1]
    model = create_model(dim, dim, N_x, density, rho, lr, input_scale, noise_in)

    # train
    train(model, train_data)

    # test
    estimated_data: list[np.ndarray] = []
    errors: list[float] = []
    errors_final: list[float] = []
    for _i, y_target in enumerate(test_data):
        model.reset_reservoir_state()
        y = model.run(y_target)
        y = np.vstack((y_target[0], y))  # insert initial value
        estimated_data.append(y)
        # Calculate errors
        error = mean_squared_error(y[:-1], y_target, root=True)
        errors.append(error)
        # Calculate errors final
        error_final = mean_squared_error(y[:-101], y_target[:-100], root=True)
        # print(f"  Mean Squared Error = {error}")
        errors_final.append(error_final)

    # plot
    if label is None:
        label = f.__name__
    if isinstance(ext, str):
        ext = [ext]
    plot2d(
        train_data,
        plot_xlim,
        plot_ylim,
        "Trajectories used for training",
        filename=f"{label}_train",
        color=color,
        ext=ext,
    )
    plot2d(
        test_data,
        plot_xlim,
        plot_ylim,
        "Trajectories used for testing",
        filename=f"{label}_test",
        color=color,
        ext=ext,
    )
    plot2d(
        estimated_data,
        plot_xlim,
        plot_ylim,
        "Autonomous running of ESN",
        filename=f"{label}_estimated",
        color=color,
        ext=ext,
    )
    save_errors(errors, label)
    save_errors(errors_final, label, "final")


def estimate_lq(
    T: float = 50,
    n_train: int = 10,
    n_test: int = 30,
    dt: float = 0.01,
    N_x: int = 300,
    density: float = 0.1,
    rho: float = 0.95,
    lr: float = 0.99,
    input_scale: float = 0.1,
    noise_in: float = 0.0,
    *,
    color: bool = False,
    ext: str | list[str] = ".png",
) -> None:
    estimate_ds_2d(
        lq,
        (1.0, 3.0, 4.0),
        xlim=(-3, 3),
        ylim=(-3, 3),
        T=T,
        n_train=n_train,
        n_test=n_test,
        dt=dt,
        N_x=N_x,
        density=density,
        rho=rho,
        lr=lr,
        input_scale=input_scale,
        noise_in=noise_in,
        label="lq",
        plot_xlim=(-3, 3),
        plot_ylim=(-3, 3),
        color=color,
        ext=ext,
    )


def estimate_vdp(
    T: float = 50,
    n_train: int = 10,
    n_test: int = 30,
    dt: float = 0.01,
    N_x: int = 300,
    density: float = 0.1,
    rho: float = 0.95,
    lr: float = 0.99,
    input_scale: float = 0.1,
    noise_in: float = 0.0,
    *,
    color: bool = False,
    ext: str | list[str] = ".png",
) -> None:
    estimate_ds_2d(
        vdp,
        (1.0,),
        xlim=(-5, 5),
        ylim=(-5, 5),
        T=T,
        n_train=n_train,
        n_test=n_test,
        dt=dt,
        N_x=N_x,
        density=density,
        rho=rho,
        lr=lr,
        input_scale=input_scale,
        noise_in=noise_in,
        label="vdp",
        plot_xlim=(-5, 5),
        plot_ylim=(-5, 5),
        color=color,
        ext=ext,
    )


def estimate_lorenz() -> None:
    T_train = 100
    T_test = 25
    dt = 0.02
    y0 = [1, 1, 1]

    ds = DynSys(lorenz, 10.0, 8.0 / 3.0, 28.0)
    data = ds(T_train + T_test, dt, y0)
    train_data, test_data = data[: int(T_train / dt)], data[int(T_train / dt) :]

    N_x = 300
    dim = train_data.shape[1]
    model = create_model(
        dim,
        dim,
        N_x,
        density=0.1,
        rho=0.95,
        leaking_rate=0.99,
        input_scale=0.1,
        noise_in=0.0,
    )
    train(model, [train_data])
    transient_data = train(model, [train_data])
    estimated_data = model.run(test_data)

    T_disp = (-15, 15)
    N_disp = (int(T_disp[0] / dt), int(T_disp[1] / dt))
    t_axis = np.arange(*T_disp, dt)
    disp_target = np.concatenate((train_data[N_disp[0] :], test_data[: N_disp[1]]))
    disp_model = np.concatenate((transient_data[N_disp[0] :], estimated_data[: N_disp[1]]))

    plt.rcParams["figure.dpi"] = 75
    plt.rcParams["font.size"] = 12
    fig = plt.figure(figsize=(7, 7))

    ax1: Any = fig.add_subplot(311)
    ax1.text(-0.15, 1, "(a)", transform=ax1.transAxes)
    ax1.text(0.2, 1.05, "Training", transform=ax1.transAxes)
    ax1.text(0.7, 1.05, "Testing", transform=ax1.transAxes)
    ax1.plot(t_axis, disp_target[:, 0], c="k", label="Target")
    ax1.plot(t_axis, disp_model[:, 0], c="gray", ls="--", label="Model")
    ax1.set_ylabel("x")
    ax1.axvline(x=0, ymin=0, ymax=1, c="k", ls=":")
    ax1.legend(bbox_to_anchor=(0, 0), loc="lower left")

    ax2: Any = fig.add_subplot(312)
    ax2.text(-0.15, 1, "(b)", transform=ax2.transAxes)
    ax2.plot(t_axis, disp_target[:, 1], c="k", label="Target")
    ax2.plot(t_axis, disp_model[:, 1], c="gray", ls="--", label="Model")
    ax2.set_ylabel("y")
    ax2.axvline(x=0, ymin=0, ymax=1, c="k", ls=":")
    ax2.legend(bbox_to_anchor=(0, 0), loc="lower left")

    ax3: Any = fig.add_subplot(313)
    ax3.text(-0.15, 1, "(c)", transform=ax3.transAxes)
    ax3.plot(t_axis, disp_target[:, 2], c="k", label="Target")
    ax3.plot(t_axis, disp_model[:, 2], c="gray", ls="--", label="Model")
    ax3.set_ylabel("z")
    ax3.set_xlabel("t")
    ax3.axvline(x=0, ymin=0, ymax=1, c="k", ls=":")
    ax3.legend(bbox_to_anchor=(0, 0), loc="lower left")


def main() -> None:
    ext = [".png", ".svg"]
    color = False
    n_test = 100
    set_seed(12345)
    # estimate_lq(color=color, n_test=n_test, ext=ext)
    estimate_vdp(color=color, n_test=n_test, ext=ext)
    # estimate_lorenz()
    plt.show()


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "env lorenz lq noqa png tikhonov txt usr vdp zlim"
# End:
