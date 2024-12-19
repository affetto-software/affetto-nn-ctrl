#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfWriter
from pyplotutil.datautil import Data

from affetto_nn_ctrl.control_utility import resolve_joints_str
from affetto_nn_ctrl.data_handling import (
    prepare_data_dir_path,
)
from affetto_nn_ctrl.event_logging import event_logger, start_logging
from affetto_nn_ctrl.plot_utility import (
    calculate_mean_err,
    extract_all_values,
    extract_common_parts,
    get_tlim_mask,
    save_figure,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.typing import ColorType

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


DEFAULT_DOF = 13


@dataclass
class TrackedTrajectoryPaths:
    model_path: Path
    active_joints: list[int]
    reference_paths: list[Path]
    reference_keys: list[str]
    motion_paths: dict[str, list[Path]]


def load_tracked_trajectory_paths(
    tracked_trajectory: str,
    given_active_joints: list[int] | None,
    reference_prefix: str,
) -> TrackedTrajectoryPaths:
    with Path(tracked_trajectory).open("rb") as f:
        tracked_trajectory_config = tomllib.load(f)
    config: dict = tracked_trajectory_config["model"]["performance"]
    model_path = Path(config["model_path"])
    active_joints = config.get("active_joints", given_active_joints)

    reference_keys: list[str] = []
    reference_paths: list[Path] = []
    for key in config:
        if key.startswith(reference_prefix):
            reference_keys.append(key)
            reference_paths.append(Path(config[key]["reference_path"]))

    motion_paths: dict[str, list[Path]] = {}
    for key in reference_keys:
        motion_paths[key] = [x["motion_path"] for x in config[key]["errors"]]

    return TrackedTrajectoryPaths(model_path, active_joints, reference_paths, reference_keys, motion_paths)


@overload
def mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    multioutput: Literal["raw_values"],
) -> np.ndarray: ...


@overload
def mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    multioutput: Literal["uniform_average"],
) -> np.floating: ...


def mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    multioutput: Literal["raw_values", "uniform_average"] = "raw_values",
) -> np.ndarray | np.floating:
    output_errors = np.average((y_true - y_pred) ** 2, axis=0)
    if multioutput == "uniform_average":
        return np.average(output_errors)
    return output_errors


@overload
def root_mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    multioutput: Literal["raw_values"],
) -> np.ndarray: ...


@overload
def root_mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    multioutput: Literal["uniform_average"],
) -> np.floating: ...


def root_mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    multioutput: Literal["raw_values", "uniform_average"] = "raw_values",
) -> np.ndarray | np.floating:
    output_errors = np.sqrt(mean_squared_error(y_true, y_pred, multioutput="raw_values"))
    if multioutput == "uniform_average":
        return np.average(output_errors)
    return output_errors


def calculate_rmse(
    reference_data: Data,
    motion_data: Data,
    active_joints: list[int],
) -> np.ndarray:
    y_true = reference_data[[f"q{x}" for x in active_joints]].to_numpy()
    y_pred = motion_data[[f"q{x}" for x in active_joints]].to_numpy()
    return root_mean_squared_error(y_true, y_pred, multioutput="raw_values")


def save_rmse(
    output_dir_path: Path,
    output_prefix: str,
    paths: TrackedTrajectoryPaths,
    all_rmse: OrderedDict[str, np.ndarray],
    ext: str = ".toml",
) -> Path:
    if not ext.startswith("."):
        ext = f".{ext}"
    output = output_dir_path / f"{output_prefix}{ext}"

    joints_str = ", ".join(map(str, paths.active_joints))
    # Calculate mean and std of all RMSE values.
    concat_rmse = np.vstack([all_rmse[key] for key in all_rmse])
    all_rmse_mean_str = ", ".join(map(str, np.mean(concat_rmse, axis=0)))
    all_rmse_std_str = ", ".join(map(str, np.std(concat_rmse, axis=0)))
    # calculate mean and std per each reference.
    rmse_mean: dict[str, np.ndarray] = {key: np.mean(all_rmse[key], axis=0) for key in all_rmse}
    rmse_std: dict[str, np.ndarray] = {key: np.std(all_rmse[key], axis=0) for key in all_rmse}
    rmse_argmax = [
        max(d, key=d.get)  # type: ignore[arg-type]
        for d in [{key: rmse_mean[key][i] for key in rmse_mean} for i in range(len(paths.active_joints))]
    ]
    rmse_argmax_str = ", ".join([f'"{key}"' for key in rmse_argmax])
    rmse_argmin = [
        min(d, key=d.get)  # type: ignore[arg-type]
        for d in [{key: rmse_mean[key][i] for key in rmse_mean} for i in range(len(paths.active_joints))]
    ]
    rmse_argmin_str = ", ".join([f'"{key}"' for key in rmse_argmin])
    text_lines = [
        "[model.performance]\n",
        f'model_path = "{paths.model_path!s}"\n',
        f"active_joints = [ {joints_str} ]\n",
        f"rmse = {{ mean = [ {all_rmse_mean_str} ], std = [ {all_rmse_std_str} ], ",
        f"argmax = [ {rmse_argmax_str} ], argmin = [ {rmse_argmin_str} ] }}\n",
        "\n",
    ]

    for reference_key, reference_path in zip(paths.reference_keys, paths.reference_paths, strict=True):
        rmse_mean_str = ", ".join(map(str, rmse_mean[reference_key]))
        rmse_std_str = ", ".join(map(str, rmse_std[reference_key]))
        rmse_argmax_str = ", ".join(map(str, np.argmax(all_rmse[reference_key], axis=0)))
        rmse_argmin_str = ", ".join(map(str, np.argmin(all_rmse[reference_key], axis=0)))
        text_lines.extend(
            [
                f"[model.performance.{reference_key}]\n",
                f'reference_path = "{reference_path!s}"\n',
                f"rmse = {{ mean = [ {rmse_mean_str} ], std = [ {rmse_std_str} ], ",
                f"argmax = [ {rmse_argmax_str} ], argmin = [ {rmse_argmin_str} ] }}\n",
                "\n",
            ],
        )

        for i, motion_path in enumerate(paths.motion_paths[reference_key]):
            rmse_str = ", ".join(map(str, all_rmse[reference_key][i]))
            text_lines.extend(
                [
                    f"[[model.performance.{reference_key}.errors]]\n",
                    f'motion_path = "{motion_path!s}"\n',
                    f"rmse = [ {rmse_str} ]\n",
                    "\n",
                ],
            )

    return output


def plot_single_motion(
    output_dir_path: Path,
    plot_prefix: str,
    active_joints: list[int],
    reference_data: Data,
    motion_data: Data,
    rmse_list: list[float],
    tshift: float,
    tlim: tuple[float, float] | None,
    *,
    dpi: str | float,
    show_legend: bool,
    ext: list[str],
    use_motion_reference: bool,
) -> list[Path]:
    all_saved_figures: list[Path] = []
    for i, joint_index in enumerate(active_joints):
        title = (
            f"Joint: {joint_index} | "
            f"Reference: {reference_data.datapath.stem} | "
            f"Motion: {motion_data.datapath.stem} | "
            f"RMSE: {rmse_list[i]} | "
            f"{output_dir_path.name}"
        )
        if use_motion_reference:
            t_ref = motion_data.t - tshift
        else:
            t_ref = reference_data.t - tshift
        t_mot = motion_data.t - tshift
        mask_ref = get_tlim_mask(t_ref, tlim)
        mask_mot = get_tlim_mask(t_mot, tlim)
        var = ("q", "dq")
        ylabels = ("Joint position [0-100]", "Joint angular velocity [0-100/s]")
        for v, ylabel in zip(var, ylabels, strict=True):
            ax: Axes
            fig: Figure
            fig, ax = plt.subplots(figsize=(18, 6))
            if use_motion_reference:
                ref = getattr(motion_data, f"{v}des{joint_index}")
            else:
                ref = getattr(reference_data, f"{v}{joint_index}")
            y = getattr(motion_data, f"{v}{joint_index}")
            (line,) = ax.plot(t_ref[mask_ref], ref[mask_ref], ls="--", label="reference trajectory")
            ax.plot(t_mot[mask_mot], y[mask_mot], c=line.get_color(), label="actual trajectory")
            ax.set_title(title)
            ax.set_xlabel("time [s]")
            ax.set_ylabel(ylabel)
            if show_legend:
                ax.legend()
            output_basename = f"{plot_prefix}_{joint_index:02d}_{v}"
            saved_figures = save_figure(
                fig,
                output_dir_path,
                f"{output_basename}",
                ext,
                loaded_from=None,
                dpi=dpi,
            )
            all_saved_figures.extend(saved_figures)
    return all_saved_figures


def merge_plot_figures(
    saved_figures: list[Path],
    *,
    prefix: str | None = None,
    suffix: str | None = None,
) -> list[Path]:
    pdf_figures = sorted(filter(lambda x: x.suffix == ".pdf", saved_figures))
    if len(pdf_figures) == 0:
        event_logger().warning("Unable to merge plots because no PDF files saved.")
        return []

    output_dirpath = extract_common_parts(*pdf_figures)
    filename_set = {x.name for x in pdf_figures}
    merged_files: list[Path] = []
    for name in filename_set:
        merger = PdfWriter()
        for pdf in filter(lambda x: x.name == name, pdf_figures):
            merger.append(pdf)
        stem = Path(name).stem
        ext = Path(name).suffix
        if prefix is not None:
            stem = f"{prefix}_{stem}"
        if suffix is not None:
            stem = f"{stem}_{suffix}"
        output = output_dirpath / f"{stem}{ext}"
        merger.write(output)
        merger.close()
        merged_files.append(output)
        event_logger().info("Saved merged plots: %s", output)
    return merged_files


def load_timeseries(dataset: list[Data], key: str, tshift: float) -> tuple[np.ndarray, np.ndarray]:
    y = extract_all_values(dataset, key)
    n = len(y[0])
    t = dataset[0].t[:n] - tshift
    return t, y


def plot_mean_err(
    ax: Axes,
    t: np.ndarray,
    y: np.ndarray,
    err_type: str | None,
    tlim: tuple[float, float] | None,
    fmt: str,
    capsize: int,
    label: str | None,
) -> Line2D:
    mask = get_tlim_mask(t, tlim)
    if err_type is None or err_type == "none":
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
    err_type: str | None,
    tlim: tuple[float, float] | None,
    color: ColorType,
    alpha: float,
) -> Axes:
    if err_type is None:
        msg = "`err_type` for `fill_between_err` must not be None."
        raise TypeError(msg)

    mask = get_tlim_mask(t, tlim)
    mean, err1, err2 = calculate_mean_err(y, err_type=err_type)
    # Note that fill_between always goes behind lines.
    if err2 is None:
        ax.fill_between(t[mask], mean[mask] + err1[mask], mean[mask] - err1[mask], facecolor=color, alpha=alpha)
    else:
        ax.fill_between(t[mask], mean[mask] + err1[mask], mean[mask] - err2[mask], facecolor=color, alpha=alpha)
    return ax


def plot_all_motions(
    output_dir_path: Path,
    plot_prefix: str,
    active_joints: list[int],
    reference_data: Data,
    motion_data_list: list[Data],
    rmse_mean_list: list[float],
    rmse_err_list: list[float],
    tshift: float,
    tlim: tuple[float, float] | None,
    *,
    dpi: str | float,
    show_legend: bool,
    ext: list[str],
    use_motion_reference: bool,
    err_type: str | None,
    fill: bool,
    fill_err_type: str | None,
    fill_alpha: float,
) -> list[Path]:
    all_saved_figures: list[Path] = []
    for i, joint_index in enumerate(active_joints):
        title = (
            f"Joint: {joint_index} | "
            f"Reference: {reference_data.datapath.stem} | "
            f"RMSE: {rmse_mean_list[i]}±{rmse_err_list[i]} | "
            f"{output_dir_path.name}"
        )
        var = ("q", "dq")
        ylabels = ("Joint position [0-100]", "Joint angular velocity [0-100/s]")
        for v, ylabel in zip(var, ylabels, strict=True):
            ax: Axes
            fig: Figure
            fig, ax = plt.subplots(figsize=(18, 6))
            t, y = load_timeseries(motion_data_list, f"{v}{joint_index}", tshift)

            t_ref = t if use_motion_reference else reference_data.t - tshift
            mask_ref = get_tlim_mask(t_ref, tlim)
            if use_motion_reference:
                y_ref = getattr(motion_data_list[0], f"{v}des{joint_index}")
            else:
                y_ref = getattr(reference_data, f"{v}{joint_index}")
            (line,) = ax.plot(t_ref[mask_ref], y_ref[mask_ref], ls="--", label="reference trajectory")

            line = plot_mean_err(ax, t, y, err_type, tlim, fmt="-", capsize=2, label="mean of actual trajectories")
            if fill:
                fill_between_err(ax, t, y, fill_err_type, tlim, line.get_color(), fill_alpha)

            ax.set_title(title)
            if v == "q":
                ax.set_ylim((-5, 105))
            ax.set_xlabel("time [s]")
            ax.set_ylabel(ylabel)
            if show_legend:
                ax.legend()
            output_basename = f"{plot_prefix}_{joint_index:02d}_{v}"
            saved_figures = save_figure(
                fig,
                output_dir_path,
                f"{output_basename}",
                ext,
                loaded_from=None,
                dpi=dpi,
            )
            all_saved_figures.extend(saved_figures)
    return all_saved_figures


def run(
    given_tracked_trajectory_path: str,
    joints_str: str | None,
    output_dir_path: Path,
    reference_prefix: str,
    output_prefix: str,
    plot_prefix: str,
    plot_ext: list[str],
    tshift: float,
    tlim: tuple[float, float] | None,
    *,
    merge_plots: bool,
    dpi: str | float,
    show_legend: bool,
    show_screen: bool | None,
    use_motion_reference: bool,
    err_type: str | None,
    fill: bool,
    fill_err_type: str | None,
    fill_alpha: float,
) -> None:
    # Resolve active joints.
    active_joints = resolve_joints_str(joints_str, dof=DEFAULT_DOF) if joints_str is not None else None
    event_logger().debug("Resolved active joints: %s", active_joints)

    # Load tracked trajectory motion file paths.
    tracked_trajectory_paths = load_tracked_trajectory_paths(
        given_tracked_trajectory_path,
        active_joints,
        reference_prefix,
    )
    active_joints = tracked_trajectory_paths.active_joints
    event_logger().debug("Loaded tracked trajectory paths from: %s", given_tracked_trajectory_path)

    # Plot reference and motion trajectories and calculate RMSE.
    all_rmse = OrderedDict[str, np.ndarray]()
    saved_figures_all_motion: list[Path] = []
    for reference_key, reference_path in zip(
        tracked_trajectory_paths.reference_keys,
        tracked_trajectory_paths.reference_paths,
        strict=True,
    ):
        reference_data = Data(reference_path)
        event_logger().debug("Loaded reference data: %s", reference_path)
        motion_data_list = [Data(motion_path) for motion_path in tracked_trajectory_paths.motion_paths[reference_key]]
        rmse_list: list[np.ndarray] = []
        saved_figures_single_motion: list[Path] = []
        for motion_data in motion_data_list:
            event_logger().debug("Loaded motion data: %s", motion_data.datapath)
            rmse = calculate_rmse(reference_data, motion_data, active_joints)
            assert rmse.ndim == 1
            assert len(rmse) == len(active_joints)
            event_logger().debug("Calculated RMSE for each joint: %s", rmse)
            rmse_list.append(rmse)
            single_motion_plot_dir = output_dir_path / reference_key / motion_data.datapath.stem
            saved_figures = plot_single_motion(
                single_motion_plot_dir,
                plot_prefix,
                active_joints,
                reference_data,
                motion_data,
                rmse.tolist(),
                tshift,
                tlim,
                dpi=dpi,
                show_legend=show_legend,
                ext=plot_ext,
                use_motion_reference=use_motion_reference,
            )
            saved_figures_single_motion.extend(saved_figures)

        if merge_plots:
            event_logger().debug("Merging the following figures: %s", saved_figures_single_motion)
            merge_plot_figures(saved_figures_single_motion, prefix="merged")
        if not show_screen:
            plt.close()

        all_motions_plot_dir = output_dir_path / reference_key / f"{output_prefix}_all"
        all_rmse[reference_key] = np.array(rmse_list)
        saved_figures = plot_all_motions(
            all_motions_plot_dir,
            f"all_{plot_prefix}",
            active_joints,
            reference_data,
            motion_data_list,
            np.mean(all_rmse[reference_key], axis=0),
            np.std(all_rmse[reference_key], axis=0),
            tshift,
            tlim,
            dpi=dpi,
            show_legend=show_legend,
            ext=plot_ext,
            use_motion_reference=use_motion_reference,
            err_type=err_type,
            fill=fill,
            fill_err_type=fill_err_type,
            fill_alpha=fill_alpha,
        )
        saved_figures_all_motion.extend(saved_figures)

    # Save calculated RMSE.
    save_rmse(output_dir_path, output_prefix, tracked_trajectory_paths, all_rmse)

    if merge_plots:
        event_logger().debug("Merging the following figures: %s", saved_figures_all_motion)
        merge_plot_figures(saved_figures_all_motion, prefix="merged")

    if show_screen:
        plt.show()


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate the root mean square error between reference and motion trajectories.",
    )
    # Configuration
    parser.add_argument(
        "tracked_trajectory",
        help="Path to a file 'tracked_trajectory.toml' to calculate RMSE.",
    )
    parser.add_argument(
        "-j",
        "--joints",
        nargs="*",
        help="Active joint index list allowing to move.",
    )
    # Input
    # Parameters
    # Output
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory where performed tracking data files are stored.",
    )
    parser.add_argument(
        "--reference-prefix",
        default="reference",
        help="Directory name prefix that will be added to reference motion trajectory.",
    )
    parser.add_argument(
        "--output-prefix",
        default="tracked_trajectory",
        help="Filename prefix that will be added to generated data files.",
    )
    parser.add_argument(
        "--plot-prefix",
        default="prediction",
        help="Filename prefix that will be added to plot figure.",
    )
    parser.add_argument("--tshift", type=float, default=0.0, help="time shift")
    parser.add_argument("--tlim", nargs="+", type=float, help="range of time")
    parser.add_argument(
        "-e",
        "--plot-ext",
        nargs="*",
        help="Filename prefix that will be added to plot figure.",
    )
    parser.add_argument(
        "--merge-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Boolean. If True, merge saved plot figures per active joint.",
    )
    parser.add_argument("--dpi", default="figure", help="Figure DPI to be saved")
    parser.add_argument(
        "--show-legend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether show legend. (default: True)",
    )
    parser.add_argument(
        "--use-motion-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Boolean. If True, use desired values in motion file instead of reference.",
    )
    parser.add_argument("-t", "--err-type", help="how to calculate errors, choose from [sd, range, se]")
    parser.add_argument(
        "--fill",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether fill between error (default: True)",
    )
    parser.add_argument("--fill-err-type", help="how to calculate errors for filling, e.g. sd, range, se")
    parser.add_argument("--fill-alpha", type=float, help="alpha value for filling")
    # Others
    parser.add_argument(
        "--show-screen",
        action=argparse.BooleanOptionalAction,
        help="Boolean. If True, show the plot figure. (default: True when test data sets are small)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Enable verbose console output. -v provides additional info. -vv provides debug output.",
    )
    return parser.parse_args()


def main() -> None:
    import sys

    args = parse()

    # Prepare input/output
    if args.output is not None:
        output_dir = Path(args.output)
    else:
        output_dir = Path(args.tracked_trajectory).parent
    start_logging(sys.argv, output_dir, __name__, args.verbose)
    event_logger().info("Output directory: %s", output_dir)
    prepare_data_dir_path(output_dir, make_latest_symlink=False)
    event_logger().debug("Parsed arguments: %s", args)
    dpi = float(args.dpi) if args.dpi != "figure" else args.dpi

    # Start mainloop
    run(
        # configuration
        args.tracked_trajectory,
        args.joints,
        # input
        # parameters
        # output
        output_dir,
        args.reference_prefix,
        args.output_prefix,
        args.plot_prefix,
        args.plot_ext,
        args.tshift,
        args.tlim,
        # boolean arguments
        merge_plots=args.merge_plots,
        dpi=dpi,
        show_legend=args.show_legend,
        show_screen=args.show_screen,
        use_motion_reference=args.use_motion_reference,
        err_type=args.err_type,
        fill=args.fill,
        fill_err_type=args.fill_err_type,
        fill_alpha=args.fill_alpha,
    )


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "arg argmax argmin des dq env pdf rb rmse sd se tlim tshift usr vv"
# End:
