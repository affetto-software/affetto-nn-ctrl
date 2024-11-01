#!/usr/bin/env python

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from pyplotutil.datautil import Data

from affetto_nn_ctrl.control_utility import resolve_joints_str
from affetto_nn_ctrl.data_handling import find_latest_data_dir_path, is_latest_data_dir_path_maybe
from affetto_nn_ctrl.event_logging import FakeLogger, get_logging_level_from_verbose_count, start_logging
from affetto_nn_ctrl.plot_utility import (
    DEFAULT_JOINT_NAMES,
    calculate_mean_err,
    event_logger,
    extract_all_values,
    extract_common_parts,
    get_tlim_mask,
    mask_data,
    pickup_datapath,
    save_figure,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.typing import ColorType


def _plot_timeseries_multi_data(
    ax: Axes,
    tlim: tuple[float, float] | None,
    dataset: list[Data],
    joint_id: int,
    key_list: Iterable[str],
    *,
    only_once: bool,
    unit: str | None,
    plot_labels: Iterable[str] | None,
) -> Axes:
    if plot_labels is None:
        if len(list(key_list)) == 1:
            plot_labels = [f"Data {i}" for i in range(len(dataset))]
        else:
            plot_labels = [f"Data {i} ({key})" for i in range(len(dataset)) for key in key_list]
    assert len(list(plot_labels)) == len(dataset) * len(list(key_list))

    labels_iter = iter(plot_labels)
    for data in dataset:
        t = data.t
        for key in key_list:
            y = getattr(data, f"{key}{joint_id}")
            if unit == "kPa":
                y = y * 600.0 / 255.0
            ax.plot(*mask_data(tlim, t, y), label=next(labels_iter))
        if only_once:
            # Plot only once since all data assumed to be the same.
            break

    return ax


def load_timeseries(dataset: list[Data], key: str, tshift: float) -> tuple[np.ndarray, np.ndarray]:
    y = extract_all_values(dataset, key)
    n = len(y[0])
    t = dataset[0].t[:n] - tshift
    return t, y


def plot_mean_err(
    ax: Axes,
    t: np.ndarray,
    y: np.ndarray,
    err_type: str,
    tlim: tuple[float, float] | None,
    fmt: str,
    capsize: int,
    label: str | None,
) -> Line2D:
    mask = get_tlim_mask(t, tlim)
    if err_type == "none":
        mean, _, _ = calculate_mean_err(y)
        lines = ax.plot(t[mask], mean[mask], fmt, label=label)
    else:
        mean, err1, err2 = calculate_mean_err(y, err_type=err_type)
        if err2 is None:
            eb = ax.errorbar(t[mask], mean[mask], yerr=err1[mask], capsize=capsize, fmt=fmt, label=label)
        else:
            eb = ax.errorbar(t[mask], mean[mask], yerr=(err1[mask], err2[mask]), capsize=capsize, fmt=fmt, label=label)
        lines = eb.lines  # type: ignore[assignment]
    return lines[0]


def fill_between_err(
    ax: Axes,
    t: np.ndarray,
    y: np.ndarray,
    err_type: str,
    tlim: tuple[float, float] | None,
    color: ColorType,
    alpha: float,
) -> Axes:
    mask = get_tlim_mask(t, tlim)
    mean, err1, err2 = calculate_mean_err(y, err_type=err_type)
    if err2 is None:
        ax.fill_between(t[mask], mean[mask] + err1[mask], mean[mask] - err1[mask], facecolor=color, alpha=alpha)
    else:
        ax.fill_between(t[mask], mean[mask] + err1[mask], mean[mask] - err2[mask], facecolor=color, alpha=alpha)
    return ax


def _plot_timeseries_mean_err(
    ax: Axes,
    tshift: float,
    tlim: tuple[float, float] | None,
    dataset: list[Data],
    joint_id: int,
    key_list: Iterable[str],
    *,
    unit: str | None,
    label: str | None,
    err_type: str,
    fill: bool,
    fill_err_type: str,
    fill_alpha: float,
) -> Axes:
    for key in key_list:
        t, y = load_timeseries(dataset, f"{key}{joint_id}", tshift)
        if unit == "kPa":
            y = y * 600.0 / 255.0
        line = plot_mean_err(ax, t, y, err_type, tlim, fmt="-", capsize=2, label=f"{label} ({key})")
        if fill:
            fill_between_err(ax, t, y, fill_err_type, tlim, line.get_color(), fill_alpha)

    return ax


def _plot_timeseries_active_joints(
    ax: Axes,
    tshift: float,
    tlim: tuple[float, float] | None,
    data: Data,
    active_joints: list[int],
    key_list: Iterable[str],
    *,
    unit: str | None,
    plot_labels: Iterable | None,
) -> Axes:
    if plot_labels is None:
        if len(list(key_list)) == 1:
            plot_labels = [f"Joint {i}" for i in range(len(active_joints))]
        else:
            plot_labels = [f"Joint {i} ({key})" for i in range(len(active_joints)) for key in key_list]
    assert len(list(plot_labels)) == len(active_joints) * len(list(key_list))

    labels_iter = iter(plot_labels)
    t = data.t - tshift
    for joint_id in active_joints:
        for key in key_list:
            y = getattr(data, f"{key}{joint_id}")
            if unit == "kPa":
                y = y * 600.0 / 255.0
            ax.plot(*mask_data(tlim, t, y), label=next(labels_iter))

    return ax


def _plot_timeseries(
    ax: Axes,
    tshift: float,
    tlim: tuple[float, float] | None,
    dataset: Data | list[Data],
    active_joints: int | list[int],
    key_list: Iterable[str],
    *,
    ylim: tuple[float, float] | None,
    ylabel: str | None,
    title: str | None,
    legend: bool,
    only_once: bool = False,
    unit: str | None = None,
    plot_labels: Iterable | None = None,
    err_type: str | None = None,
    fill: bool = True,
    fill_err_type: str = "range",
    fill_alpha: float = 0.4,
) -> Axes:
    if not isinstance(active_joints, list):
        if isinstance(dataset, Data):
            dataset = [dataset]
        if err_type is None:
            _plot_timeseries_multi_data(
                ax,
                tlim,
                dataset,
                active_joints,  # int, not list[int]
                key_list,
                only_once=only_once,
                unit=unit,
                plot_labels=plot_labels,
            )
        else:
            _plot_timeseries_mean_err(
                ax,
                tshift,
                tlim,
                dataset,
                active_joints,  # int, not list[int]
                key_list,
                unit=unit,
                label=None,
                err_type=err_type,
                fill=fill,
                fill_err_type=fill_err_type,
                fill_alpha=fill_alpha,
            )
    else:
        if isinstance(dataset, list):
            dataset = dataset[0]
            msg = "Unable to plot multiple data with multiple joints simultaneously."
            warnings.warn(msg, stacklevel=2)
        _plot_timeseries_active_joints(
            ax,
            tshift,
            tlim,
            dataset,  # Data, not list[Data]
            active_joints,
            key_list,
            unit=unit,
            plot_labels=plot_labels,
        )

    ax.grid(axis="y", visible=True)
    if ylabel is None:
        ylabel = ", ".join(key_list)
    if unit == "kPa":
        ylabel += " [kPa]"
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    if legend:
        ax.legend()
    if title:
        ax.set_title(title)

    return ax


def plot_pressure_command(
    ax: Axes,
    tshift: float,
    tlim: tuple[float, float] | None,
    dataset: Data | list[Data],
    active_joints: int | list[int],
    *,
    ylim: tuple[float, float] | None = None,
    legend: bool = False,
    unit: str = "kPa",
    only_once: bool = True,
) -> None:
    title = "Pressure at controllable valve"
    if ylim is None:
        ylim = (-50, 650)
    _plot_timeseries(
        ax,
        tshift,
        tlim,
        dataset,
        active_joints,
        ("ca", "cb"),
        ylim=ylim,
        ylabel="Pressure",
        title=title,
        legend=legend,
        only_once=only_once,
        unit=unit,
    )


def plot_pressure_sensor(
    ax: Axes,
    tshift: float,
    tlim: tuple[float, float] | None,
    dataset: Data | list[Data],
    active_joints: int | list[int],
    *,
    ylim: tuple[float, float] | None = None,
    legend: bool = False,
    err_type: str | None = None,
    fill: bool = True,
    fill_err_type: str = "range",
    fill_alpha: float = 0.4,
) -> None:
    title = "Pressure in actuator chamber"
    _plot_timeseries(
        ax,
        tshift,
        tlim,
        dataset,
        active_joints,
        ("pa", "pb"),
        ylim=ylim,
        ylabel="Pressure [kPa]",
        title=title,
        legend=legend,
        err_type=err_type,
        fill=fill,
        fill_err_type=fill_err_type,
        fill_alpha=fill_alpha,
    )


def plot_velocity(
    ax: Axes,
    tshift: float,
    tlim: tuple[float, float] | None,
    dataset: Data | list[Data],
    active_joints: int | list[int],
    *,
    ylim: tuple[float, float] | None = None,
    legend: bool = False,
    err_type: str | None = None,
    fill: bool = True,
    fill_err_type: str = "range",
    fill_alpha: float = 0.4,
) -> None:
    title = "Joint angle velocity"
    _plot_timeseries(
        ax,
        tshift,
        tlim,
        dataset,
        active_joints,
        ("dq",),
        ylim=ylim,
        ylabel="Velocity [0-100/s]",
        title=title,
        legend=legend,
        err_type=err_type,
        fill=fill,
        fill_err_type=fill_err_type,
        fill_alpha=fill_alpha,
    )


def plot_position(
    ax: Axes,
    tshift: float,
    tlim: tuple[float, float] | None,
    dataset: Data | list[Data],
    active_joints: int | list[int],
    *,
    ylim: tuple[float, float] | None = None,
    legend: bool = False,
    err_type: str | None = None,
    fill: bool = True,
    fill_err_type: str = "range",
    fill_alpha: float = 0.4,
) -> None:
    title = "Joint angle"
    if ylim is None:
        ylim = (-10, 110)
    _plot_timeseries(
        ax,
        tshift,
        tlim,
        dataset,
        active_joints,
        ("q",),
        ylim=ylim,
        ylabel="Position [0-100]",
        title=title,
        legend=legend,
        err_type=err_type,
        fill=fill,
        fill_err_type=fill_err_type,
        fill_alpha=fill_alpha,
    )


def plot_multi_data(
    datapath_list: list[Path],
    joint_id: int,
    plot_keys: str,
    *,
    sharex: bool = False,
    tshift: float = 0.0,
    tlim: tuple[float, float] | None = None,
    title: str | None = None,
    legend: bool = False,
    show_cmd_once: bool = True,
    err_type: str | None = None,
    fill: bool = True,
    fill_err_type: str = "range",
    fill_alpha: float = 0.4,
) -> tuple[Figure, list[Axes]]:
    n_keys = len(plot_keys)
    figsize = (16, 4 * n_keys)
    fig, axes = plt.subplots(nrows=n_keys, sharex=sharex, figsize=figsize)

    if n_keys == 1:
        axes = [axes]
    dataset = [Data(csv) for csv in datapath_list]
    for ax, v in zip(axes, plot_keys, strict=True):
        if v == "c":
            plot_pressure_command(
                ax,
                tshift,
                tlim,
                dataset,
                joint_id,
                ylim=None,
                legend=legend,
                unit="kPa",
                only_once=show_cmd_once,
            )
        elif v == "p":
            plot_pressure_sensor(
                ax,
                tshift,
                tlim,
                dataset,
                joint_id,
                ylim=None,
                legend=legend,
                err_type=err_type,
                fill=fill,
                fill_err_type=fill_err_type,
                fill_alpha=fill_alpha,
            )
        elif v == "v":
            plot_velocity(
                ax,
                tshift,
                tlim,
                dataset,
                joint_id,
                ylim=None,
                legend=legend,
                err_type=err_type,
                fill=fill,
                fill_err_type=fill_err_type,
                fill_alpha=fill_alpha,
            )
        elif v == "q":
            plot_position(
                ax,
                tshift,
                tlim,
                dataset,
                joint_id,
                ylim=None,
                legend=legend,
                err_type=err_type,
                fill=fill,
                fill_err_type=fill_err_type,
                fill_alpha=fill_alpha,
            )
        else:
            msg = f"unrecognized plot variable: {v}"
            raise ValueError(msg)

    axes[-1].set_xlabel("time [s]")
    if title:
        fig.suptitle(title)

    return fig, axes


def plot_multi_joint(
    datapath: Path,
    active_joints: list[int],
    plot_keys: str,
    *,
    sharex: bool = False,
    tshift: float = 0.0,
    tlim: tuple[float, float] | None = None,
    title: str | None = None,
    legend: bool = False,
    show_cmd_once: bool = True,
) -> tuple[Figure, list[Axes]]:
    n_keys = len(plot_keys)
    figsize = (8, 4 * n_keys)
    fig, axes = plt.subplots(nrows=n_keys, sharex=sharex, figsize=figsize)

    if n_keys == 1:
        axes = [axes]
    data = Data(datapath)
    for ax, v in zip(axes, plot_keys, strict=True):
        if v == "c":
            plot_pressure_command(
                ax,
                tshift,
                tlim,
                data,
                active_joints,
                ylim=None,
                legend=legend,
                unit="kPa",
                only_once=show_cmd_once,
            )
        elif v == "p":
            plot_pressure_sensor(
                ax,
                tshift,
                tlim,
                data,
                active_joints,
                ylim=None,
                legend=legend,
            )
        elif v == "v":
            plot_velocity(
                ax,
                tshift,
                tlim,
                data,
                active_joints,
                ylim=None,
                legend=legend,
            )
        elif v == "q":
            plot_position(
                ax,
                tshift,
                tlim,
                data,
                active_joints,
                ylim=None,
                legend=legend,
            )
        else:
            msg = f"unrecognized plot variable: {v}"
            raise ValueError(msg)

    axes[-1].set_xlabel("time [s]")
    if title:
        fig.suptitle(title)

    return fig, axes


def _plot_data_across_multi_joints(
    datapath: Path,
    active_joints: list[int],
    plot_keys: str,
    *,
    sharex: bool,
    tshift: float,
    tlim: tuple[float, float] | None,
    title: str | None,
    legend: bool,
    show_cmd_once: bool,
    savefig_dir: Path,
    ext_list: list[str] | None,
    dpi: float | str,
) -> None:
    savefig_basename = f"{plot_keys}/multi_joints/"
    if title is None:
        savefig_basename += "_".join(map(str, active_joints))
        title = f"Joints: {' '.join(map(str, active_joints))}"
    else:
        savefig_basename += title.lower()

    fig, _ = plot_multi_joint(
        datapath,
        active_joints,
        plot_keys,
        sharex=sharex,
        tshift=tshift,
        tlim=tlim,
        title=title,
        legend=legend,
        show_cmd_once=show_cmd_once,
    )
    save_figure(fig, savefig_dir, savefig_basename, ext_list, dpi=dpi)


def _plot_specific_joint_across_multi_data(
    datapath_list: list[Path],
    active_joints: list[int],
    joint_name: str | None,
    plot_keys: str,
    *,
    sharex: bool,
    tshift: float,
    tlim: tuple[float, float] | None,
    title: str | None,
    legend: bool,
    show_cmd_once: bool,
    savefig_dir: Path,
    ext_list: list[str] | None,
    dpi: float | str,
    err_type: str | None,
    fill: bool,
    fill_err_type: str | None,
    fill_alpha: float | None,
) -> None:
    if len(active_joints) > 1:
        msg = "Multiple data plots with multiple joints are not supported."
        event_logger().warning(msg)
        warnings.warn(msg, stacklevel=2)
    joint_id = active_joints[0]
    if joint_name is None:
        joint_name = DEFAULT_JOINT_NAMES.get(str(joint_id), "unknown_joint")

    savefig_basename = f"{plot_keys}/"
    if err_type:
        savefig_basename += "mean_err/"
    else:
        savefig_basename += "multi_data/"
    if title is None:
        savefig_basename += f"{joint_id:02}_{joint_name}"
        title = f"{joint_id:02}: {joint_name}"
    else:
        savefig_basename += title.lower()

    if fill_err_type is None:
        fill_err_type = "range"
    if fill_alpha is None:
        fill_alpha = 0.4

    fig, _ = plot_multi_data(
        datapath_list,
        joint_id,
        plot_keys,
        sharex=sharex,
        tshift=tshift,
        tlim=tlim,
        title=title,
        legend=legend,
        show_cmd_once=show_cmd_once,
        err_type=err_type,
        fill=fill,
        fill_err_type=fill_err_type,
        fill_alpha=fill_alpha,
    )
    save_figure(fig, savefig_dir, savefig_basename, ext_list, dpi=dpi)


def collect_datapath(
    given_datapath: list[str],
    pickup_list: list[int | str] | None,
    active_joints: list[int],
    *,
    latest: bool,
) -> tuple[Path | list[Path], Path]:
    if len(active_joints) > 1 or (len(given_datapath) == 1 and Path(given_datapath[0]).is_file()):
        if len(given_datapath) > 1:
            msg = "Multiple data plots with multiple joints are not supported."
            event_logger().warning(msg)
            warnings.warn(msg, stacklevel=2)
        datapath = Path(given_datapath[0])
        if datapath.is_dir():
            msg = f"{datapath} is a directory. Specify a CSV file."
            raise ValueError(msg)
        event_logger().info("Loading single CSV file: %s", datapath)
        default_savefig_dir = datapath.parent / datapath.stem
        event_logger().debug("Default save directory: %s", default_savefig_dir)
        return datapath, default_savefig_dir

    if len(given_datapath) == 0:
        msg = "No datapath provided"
        raise ValueError(msg)

    if len(given_datapath) == 1:
        dirpath = Path(given_datapath[0])
        if not dirpath.is_dir():
            msg = f"{dirpath} is a file. Specify a directory."
            raise ValueError(msg)
        if latest and not is_latest_data_dir_path_maybe(dirpath):
            dirpath = find_latest_data_dir_path(dirpath)
        event_logger().info("Collecting CSV files in '%s'...", dirpath)
        datapath_list = sorted(dirpath.glob("*.csv"), key=lambda path: path.name)
        event_logger().info(" %s files found.", len(datapath_list))
        event_logger().debug("Default save directory: %s", dirpath)

    else:
        if any(Path(path).is_dir() for path in given_datapath):
            msg = f"Unable to provide multiple directories: {given_datapath}"
            raise ValueError(msg)
        datapath_list = [Path(p) for p in given_datapath]
        event_logger().info("Provided %s data files.", len(datapath_list))
        dirpath = extract_common_parts(*datapath_list)
        event_logger().debug("Default save directory: %s", dirpath)

    if pickup_list is not None:
        datapath_list = pickup_datapath(datapath_list, pickup_list)
        event_logger().info("Only specified data will be loaded: %s", {",".join(map(str, pickup_list))})

    return datapath_list, dirpath


def plot(
    datapath: Path | list[Path],
    plot_keys: str,
    active_joints: list[int],
    joint_name: str | None,
    output_dir: Path,
    *,
    sharex: bool,
    tshift: float,
    tlim: tuple[float, float] | None,
    title: str | None,
    show_legend: bool,
    show_cmd_once: bool,
    ext_list: list[str] | None,
    dpi: float | str,
    err_type: str | None,
    fill: bool,
    fill_err_type: str | None,
    fill_alpha: float | None,
) -> None:
    if isinstance(datapath, Path):
        _plot_data_across_multi_joints(
            datapath,
            active_joints,
            plot_keys,
            sharex=sharex,
            tshift=tshift,
            tlim=tlim,
            title=title,
            legend=show_legend,
            show_cmd_once=show_cmd_once,
            savefig_dir=output_dir,
            ext_list=ext_list,
            dpi=dpi,
        )
    else:
        _plot_specific_joint_across_multi_data(
            datapath,
            active_joints,
            joint_name,
            plot_keys,
            sharex=sharex,
            tshift=tshift,
            tlim=tlim,
            title=title,
            legend=show_legend,
            show_cmd_once=show_cmd_once,
            savefig_dir=output_dir,
            ext_list=ext_list,
            dpi=dpi,
            err_type=err_type,
            fill=fill,
            fill_err_type=fill_err_type,
            fill_alpha=fill_alpha,
        )


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot script to visualize time series data")
    parser.add_argument("datapath", nargs="+", help="list of paths to data file or directory")
    parser.add_argument("--pickup", nargs="+", help="pick up specified indices of data files")
    parser.add_argument(
        "-k",
        "--plot-keys",
        default="cpvq",
        help="string representing variable to plot consisting of 'c', 'p', 'v' and 'q'",
    )
    parser.add_argument(
        "-j",
        "--joints",
        nargs="*",
        help="Active joint index list allowing to move.",
    )
    parser.add_argument("--joint-name", help="joint name")
    parser.add_argument("-t", "--err-type", help="how to calculate errors, choose from [sd, range, se]")
    parser.add_argument(
        "-l",
        "--latest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether try to find latest directory or not (default: True)",
    )
    parser.add_argument("--title", help="figure title")
    parser.add_argument(
        "--sharex",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether share x axis (default: True)",
    )
    parser.add_argument("--tshift", type=float, default=0.0, help="time shift")
    parser.add_argument("--tlim", nargs="+", type=float, help="range of time")
    parser.add_argument(
        "-e",
        "--save-ext",
        dest="ext",
        nargs="*",
        help="extensions that the figure will be saved as",
    )
    parser.add_argument("--dpi", default="figure", help="figure DPI to be saved")
    parser.add_argument("-o", "--output-dir", help="path to directory that figures are saved")
    parser.add_argument(
        "--show-screen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether plot screen is shown or not (default: True)",
    )
    parser.add_argument(
        "--show-cmd-once",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether show command data only once (default: True)",
    )
    parser.add_argument(
        "--show-legend",
        action=argparse.BooleanOptionalAction,
        help=f"whether show legend (default: True when --joints < {DEFAULT_SHOW_LEGEND_N_JOINTS})",
    )
    parser.add_argument(
        "--fill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether fill between error (default: True)",
    )
    parser.add_argument("--fill-err-type", help="how to calculate errors for filling, e.g. sd, range, se")
    parser.add_argument("--fill-alpha", type=float, help="alpha value for filling")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Enable verbose console output. -v provides additional info. -vv provides debug output.",
    )
    return parser.parse_args()


DEFAULT_SHOW_LEGEND_N_JOINTS = 4
DEFAULT_DOF = 13


def main() -> None:
    import sys

    args = parse()
    if args.tlim is not None:
        if len(args.tlim) == 1:
            args.tlim = (0, args.tlim[0])
        else:
            args.tlim = (min(args.tlim), max(args.tlim))
    dpi = float(args.dpi) if args.dpi != "figure" else args.dpi
    if args.verbose > 0 and isinstance(event_logger(), FakeLogger):
        event_logger().setLevel(get_logging_level_from_verbose_count(args.verbose))

    active_joints = resolve_joints_str(args.joints, DEFAULT_DOF)
    if args.show_legend is None:
        args.show_legend = bool(len(active_joints) < DEFAULT_SHOW_LEGEND_N_JOINTS)
    datapath_list, default_output_dir = collect_datapath(args.datapath, args.pickup, active_joints, latest=args.latest)
    output_dir_path = Path(args.output_dir) if args.output_dir is not None else default_output_dir
    if args.ext is not None and len(args.ext) > 0:
        start_logging(sys.argv, output_dir_path, __name__, args.verbose)

    plot(
        datapath_list,
        plot_keys=args.plot_keys,
        active_joints=active_joints,
        joint_name=args.joint_name,
        output_dir=output_dir_path,
        sharex=args.sharex,
        tshift=args.tshift,
        tlim=args.tlim,
        title=args.title,
        show_legend=args.show_legend,
        show_cmd_once=args.show_cmd_once,
        ext_list=args.ext,
        dpi=dpi,
        err_type=args.err_type,
        fill=args.fill,
        fill_err_type=args.fill_err_type,
        fill_alpha=args.fill_alpha,
    )

    if args.show_screen or args.show_screen is None:
        plt.show()


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "cb cmd cpvq csv datapath dir dq env pb png savefig sd se sharex tlim tshift usr vv"
# End:
