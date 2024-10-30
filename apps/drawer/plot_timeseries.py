#!/usr/bin/env python

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import matplotlib.pyplot as plt
import numpy as np
from pyplotutil.datautil import Data

from .plot_utility import (
    DEFAULT_JOINT_NAMES,
    calculate_mean_err,
    extract_all_values,
    extract_common_parts,
    get_tlim_mask,
    mask_data,
    savefig,
)

if TYPE_CHECKING:
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
    only_once: bool = False,
    unit: str | None = None,
    plot_labels: Iterable[str] | None = None,
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
        lines = eb.lines
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
    err_type: str,
    *,
    unit: str | None = None,
    label: str | None = None,
    fill: bool = True,
    fill_err_type: str = "range",
    fill_alpha: float = 0.4,
) -> Line2D:
    for key in key_list:
        t, y = load_timeseries(dataset, f"{key}{joint_id}", tshift)
        if unit == "kPa":
            y = y * 600.0 / 255.0
        line = plot_mean_err(ax, t, y, err_type, tlim, fmt="-", capsize=2, label=f"{label} ({key})")
        if fill:
            fill_between_err(ax, t, y, fill_err_type, tlim, line.get_color(), fill_alpha)

    return line


def _plot_timeseries_active_joints(
    ax: Axes,
    tlim: tuple[float, float] | None,
    data: Data,
    active_joints: list[int],
    key_list: Iterable[str],
    *,
    unit: str | None = None,
    plot_labels: Iterable | None = None,
) -> Axes:
    if plot_labels is None:
        if len(list(key_list)) == 1:
            plot_labels = [f"Joint {i}" for i in range(len(active_joints))]
        else:
            plot_labels = [f"Joint {i} ({key})" for i in range(len(active_joints)) for key in key_list]
    assert len(list(plot_labels)) == len(active_joints) * len(list(key_list))

    labels_iter = iter(plot_labels)
    t = data.t
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
    joint_id_list: int | list[int],
    key_list: Iterable[str],
    err_type: str | None,
    *,
    only_once: bool = False,
    unit: str | None = None,
    plot_labels: Iterable | None = None,
    ylim: tuple[float, float] | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    legend: bool = False,
    fill: bool = True,
    fill_err_type: str = "range",
    fill_alpha: float = 0.4,
) -> Axes:
    if not isinstance(joint_id_list, list):
        if isinstance(dataset, Data):
            dataset = [dataset]
        if err_type is None:
            _plot_timeseries_multi_data(
                ax,
                tlim,
                dataset,
                joint_id_list,  # int, not list[int]
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
                joint_id_list,  # int, not list[int]
                key_list,
                err_type,
                unit=unit,
                label=None,
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
            tlim,
            dataset,  # Data, not list[Data]
            joint_id_list,
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
    joint_id_list: int | list[int],
    *,
    unit: str = "kPa",
    only_once: bool = False,
    ylim: tuple[float, float] | None = None,
    legend: bool = False,
) -> None:
    title = "Pressure at controllable valve"
    # if ylim is None:
    #     ylim = (-50, 650)
    _plot_timeseries(
        ax,
        tshift,
        tlim,
        dataset,
        joint_id_list,
        ("ca", "cb"),
        err_type=None,
        unit=unit,
        only_once=only_once,
        ylim=ylim,
        ylabel="Pressure",
        legend=legend,
        title=title,
    )


def plot_pressure_sensor(
    ax: Axes,
    tshift: float,
    tlim: tuple[float, float] | None,
    dataset: Data | list[Data],
    joint_id_list: int | list[int],
    err_type: str | None,
    ylim: tuple[float, float] | None = None,
    legend: bool = False,
    fill: bool = True,
    fill_err_type: str = "range",
    fill_alpha: float = 0.4,
) -> None:
    title = "Pressure in actuator chamber"
    # if ylim is None:
    #     ylim = (-50, 650)
    _plot_timeseries(
        ax,
        tshift,
        tlim,
        dataset,
        joint_id_list,
        ("pa", "pb"),
        err_type,
        ylim=ylim,
        ylabel="Pressure [kPa]",
        legend=legend,
        title=title,
        fill=fill,
        fill_err_type=fill_err_type,
        fill_alpha=fill_alpha,
    )


def plot_velocity(
    ax: Axes,
    tshift: float,
    tlim: tuple[float, float] | None,
    dataset: Data | list[Data],
    joint_id_list: int | list[int],
    err_type: str | None,
    ylim: tuple[float, float] | None = None,
    legend: bool = False,
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
        joint_id_list,
        ("dq",),
        err_type,
        ylim=ylim,
        ylabel="Velocity [0-100/s]",
        legend=legend,
        title=title,
        fill=fill,
        fill_err_type=fill_err_type,
        fill_alpha=fill_alpha,
    )


def plot_position(
    ax: Axes,
    tshift: float,
    tlim: tuple[float, float] | None,
    dataset: Data | list[Data],
    joint_id_list: int | list[int],
    err_type: str | None,
    ylim: tuple[float, float] | None = None,
    legend: bool = False,
    fill: bool = True,
    fill_err_type: str = "range",
    fill_alpha: float = 0.4,
) -> None:
    title = "Joint angle"
    # if ylim is None:
    #     ylim = (-10, 110)
    _plot_timeseries(
        ax,
        tshift,
        tlim,
        dataset,
        joint_id_list,
        ("q",),
        err_type,
        ylim=ylim,
        ylabel="Position [0-100]",
        legend=legend,
        title=title,
        fill=fill,
        fill_err_type=fill_err_type,
        fill_alpha=fill_alpha,
    )


def plot_multi_data(
    datapath_list: list[Path],
    joint_id: int,
    plot_vars: str,
    err_type: str | None,
    sharex: bool = False,
    tshift: float = 0.0,
    tlim: tuple[float, float] | None = None,
    legend: bool = False,
    title: str | None = None,
    only_once: bool = False,
    fill: bool = True,
    fill_err_type: str = "range",
    fill_alpha: float = 0.4,
) -> tuple[Figure, Axes]:
    n_vars = len(plot_vars)
    figsize = (16, 4 * n_vars)
    fig, axes = plt.subplots(nrows=n_vars, sharex=sharex, figsize=figsize)

    if n_vars == 1:
        axes = [axes]
    dataset = [Data(csv) for csv in datapath_list]
    for ax, v in zip(axes, plot_vars):
        if v == "c":
            plot_pressure_command(
                ax,
                tshift,
                tlim,
                dataset,
                joint_id,
                unit="kPa",
                only_once=only_once,
                ylim=None,
                legend=legend,
            )
        elif v == "p":
            plot_pressure_sensor(
                ax,
                tshift,
                tlim,
                dataset,
                joint_id,
                err_type,
                ylim=None,
                legend=legend,
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
                err_type,
                ylim=None,
                legend=legend,
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
                err_type,
                ylim=None,
                legend=legend,
                fill=fill,
                fill_err_type=fill_err_type,
                fill_alpha=fill_alpha,
            )
        else:
            raise ValueError(f"unrecognized plot variable: {v}")

    axes[-1].set_xlabel("time [s]")
    if title:
        fig.suptitle(title)

    return fig, ax


def plot_multi_joint(
    datapath: Path,
    joint_id_list: list[int],
    plot_vars: str,
    sharex: bool = False,
    tshift: float = 0.0,
    tlim: tuple[float, float] | None = None,
    legend: bool = False,
    title: str | None = None,
    only_once: bool = False,
) -> tuple[Figure, Axes]:
    n_vars = len(plot_vars)
    figsize = (8, 4 * n_vars)
    fig, axes = plt.subplots(nrows=n_vars, sharex=sharex, figsize=figsize)

    data = Data(datapath)
    for ax, v in zip(axes, plot_vars):
        if v == "c":
            plot_pressure_command(
                ax, tshift, tlim, data, joint_id_list, unit="kPa", only_once=only_once, ylim=None, legend=legend
            )
        elif v == "p":
            plot_pressure_sensor(ax, tshift, tlim, data, joint_id_list, err_type=None, ylim=None, legend=legend)
        elif v == "v":
            plot_velocity(ax, tshift, tlim, data, joint_id_list, err_type=None, ylim=None, legend=legend)
        elif v == "q":
            plot_position(ax, tshift, tlim, data, joint_id_list, err_type=None, ylim=None, legend=legend)
        else:
            raise ValueError(f"unrecognized plot variable: {v}")

    axes[-1].set_xlabel("time [s]")
    if title:
        fig.suptitle(title)

    return fig, ax


def save_figure(fig: Figure, savedir: Path, basename: str, ext: list[str] | None) -> Path:
    filename = savedir / basename
    fig.tight_layout()
    if ext is not None:
        savefig(fig, filename, ext)
    return filename


def plot(
    datapath_list: list[str],
    pickup_list: list[int | str] | None,
    plot_vars: str | None,
    joint_id_list: list[int] | None,
    joint_name: str | None,
    chamber: str | None,
    pose: str | None,
    latest: bool,
    tshift: float,
    tlim: tuple[float, float] | None,
    title: str | None,
    err_type: str | None,
    ext: list[str] | None,
    savefig_dir: str | None,
    only_once: bool,
    fill: bool,
    fill_err_type: str | None,
    fill_alpha: float | None,
) -> None:
    if plot_vars is None:
        plot_vars = "cpvq"

    if joint_id_list is not None and len(joint_id_list) > 1:
        if len(datapath_list) > 1:
            warnings.warn(f"Multiple data plots with multiple joints are not supported.")
        datapath = Path(datapath_list[0])
        if datapath.is_dir():
            raise ValueError(f"{datapath} is a directory. Specify a CSV file.")
        if savefig_dir is not None:
            savefig_dir_path = Path(savefig_dir)
        else:
            savefig_dir_path = datapath

        if title is None:
            savefig_basename = f"{plot_vars.replace('+', 'x')}/"
            savefig_basename += f"multi_joint/"
            savefig_basename += "_".join(map(str, joint_id_list))
            title = f"Joints: {' '.join(map(str, joint_id_list))}"
        else:
            savefig_basename = f"{plot_vars.replace('+', 'x')}/"
            savefig_basename += f"multi_joint/"
            savefig_basename += title.translate(str.maketrans({":": "", " ": "_", "(": "", ")": ""})).lower()
        legend = False
        if len(joint_id_list) <= 4:
            legend = True
        fig, _ = plot_multi_joint(
            datapath,
            joint_id_list,
            plot_vars,
            sharex=True,
            tshift=tshift,
            tlim=tlim,
            legend=legend,
            title=title,
            only_once=only_once,
        )
        savefig_path = save_figure(fig, savefig_dir_path, savefig_basename, ext)

    else:
        if len(datapath_list) == 1:
            dirpath = Path(datapath_list[0])
            if not dirpath.is_dir():
                raise ValueError(f"{dirpath} is a file. Specify a directory or joint list.")
            if chamber is not None and pose is not None:
                dirpath = dirpath / chamber / pose
            if latest:
                if not is_latest_dir_maybe(dirpath):
                    dirpath = find_latest_dir(dirpath)
            print(f"Collecting CSV files in '{dirpath}'...", end="")
            args = sorted(dirpath.glob("*.csv"), key=lambda path: path.name)
            print(f" {len(args)} files found.")

        elif len(datapath_list) > 1:
            for path in datapath_list:
                if Path(path).is_dir():
                    raise ValueError(f"Unable to provide multiple directories: {datapath_list}")
            dirpath = extract_common_parts(*datapath_list)
            args = [Path(p) for p in datapath_list]

        else:
            raise ValueError(f"No datapath provided")

        if pickup_list is not None:
            args = pickup_paths(args, pickup_list)
            print(f"Only specified indices will be used: {','.join(map(str, pickup_list))}")

        if joint_id_list is None:
            joint_id, joint_name = find_joint_index_name(dirpath)
        else:
            if len(joint_id_list) > 1:
                warnings.warn(f"Multiple data plots with multiple joints are not supported.")
            joint_id = joint_id_list[0]
        if joint_name is None:
            joint_name = DEFAULT_JOINT_NAMES.get(str(joint_id), "unknown_joint")

        if savefig_dir is not None:
            savefig_dir_path = Path(savefig_dir)
        else:
            savefig_dir_path = dirpath

        if title is None:
            savefig_basename = f"{plot_vars.replace('+', 'x')}/"
            if err_type:
                savefig_basename += f"mean_err/"
            else:
                savefig_basename += f"multi_data/"
            savefig_basename += f"{joint_id:02}_{joint_name}"
            title = f"{joint_id:02}: {joint_name}"
            if chamber is None:
                try:
                    chamber = find_active_chamber_name(dirpath)
                    pose = find_pose_name(dirpath)
                except ValueError:
                    pass
                else:
                    title += f" (chamber: {chamber[1].upper()}, {pose})"
        else:
            savefig_basename = f"{plot_vars.replace('+', 'x')}/"
            if err_type:
                savefig_basename += f"mean_err/"
            else:
                savefig_basename += f"multi_data/"
            savefig_basename += title.translate(str.maketrans({":": "", " ": "_", "(": "", ")": ""})).lower()

        if fill_err_type is None:
            fill_err_type = "range"
        if fill_alpha is None:
            fill_alpha = 0.4

        if len(args) == 0:
            raise RuntimeError(f"No data found: datapath: {datapath_list} joint ID: {joint_id_list}")
        fig, _ = plot_multi_data(
            args,
            joint_id,
            plot_vars,
            err_type,
            sharex=True,
            tshift=tshift,
            tlim=tlim,
            legend=False,
            title=title,
            only_once=only_once,
            fill=fill,
            fill_err_type=fill_err_type,
            fill_alpha=fill_alpha,
        )
        savefig_path = save_figure(fig, savefig_dir_path, savefig_basename, ext)

    # for debugging purpose
    _ = savefig_path
    # print(f"Saving to: {savefig_path}")


def parse():
    parser = argparse.ArgumentParser(description="Plot script to visualize time series data")
    parser.add_argument("datapath", nargs="+", help="list of paths to data file or directory")
    parser.add_argument("--pickup", nargs="+", help="pick up specified indices of data files")
    parser.add_argument(
        "-v",
        "--plot-vars",
        help="string repreesenting variable to plot consisting of 'c', 'p', 'v' and 'q'",
    )
    parser.add_argument("-j", "--joints", nargs="+", type=int, help="list of joint ids")
    parser.add_argument("--joint-name", help="joint name")
    parser.add_argument("-c", "--chamber", help="active chamber side, ca or cb")
    parser.add_argument("-p", "--pose", help="pose name, pose0 or pose1")
    parser.add_argument("-t", "--err-type", help="how to calculate errors, e.g. sd, range, se")
    parser.add_argument(
        "-l",
        "--latest",
        action=argparse.BooleanOptionalAction,
        help="whether try to find latest directory or not",
    )
    parser.add_argument("--title", help="figure title")
    parser.add_argument(
        "--all-joints",
        action=argparse.BooleanOptionalAction,
        help="if specified plot all joints",
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
    parser.add_argument("-o", "--output-dir", help="path to directory that firegures are saved")
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        help="whether plot screen is shown or not",
    )
    parser.add_argument(
        "--command-only-once",
        action=argparse.BooleanOptionalAction,
        help="whether plot command only once (default: True)",
    )
    parser.add_argument(
        "--fill",
        action=argparse.BooleanOptionalAction,
        help="whether fill between error (default: True)",
    )
    parser.add_argument("--fill-err-type", help="how to calculate errors for filling, e.g. sd, range, se")
    parser.add_argument("--fill-alpha", type=float, help="alpha value for filling")
    return parser.parse_args()


def main():
    args = parse()
    if args.latest is None:
        args.latest = True
    if args.tlim is not None:
        if len(args.tlim) == 1:
            args.tlim = (0, args.tlim[0])
        else:
            args.tlim = (min(args.tlim), max(args.tlim))
    if args.all_joints:
        args.joints = list(range(13))
        if args.title is None:
            args.title = "Whole body"
    if args.command_only_once is None:
        args.command_only_once = True
    if args.fill is None:
        args.fill = True

    plot(
        args.datapath,
        pickup_list=args.pickup,
        plot_vars=args.plot_vars,
        joint_id_list=args.joints,
        joint_name=args.joint_name,
        chamber=args.chamber,
        pose=args.pose,
        latest=args.latest,
        tshift=args.tshift,
        tlim=args.tlim,
        title=args.title,
        err_type=args.err_type,
        ext=args.ext,
        savefig_dir=args.output_dir,
        only_once=args.command_only_once,
        fill=args.fill,
        fill_err_type=args.fill_err_type,
        fill_alpha=args.fill_alpha,
    )

    if args.show or args.show is None:
        plt.show()


if __name__ == "__main__":
    main()
