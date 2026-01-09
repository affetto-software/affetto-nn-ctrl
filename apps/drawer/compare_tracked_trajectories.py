#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
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
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.typing import ColorType

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[reportMissingImports]


DEFAULT_DOF = 13


@dataclass
class RmseData:
    tracked_trajectory_path: Path
    model_path: Path | None
    active_joints: list[int]
    reference_paths: list[Path]
    reference_keys: list[str]
    motion_paths: dict[str, list[Path]]  # ref_key -> list[motion path]

    all_rmse_mean_list: list[float]  # all mean RMSE for each joint
    all_rmse_std_list: list[float]  # all std RMSE for each joint

    rmse_mean_list: dict[str, list[float]]  # ref_key -> mean RMSE for each joint
    rmse_std_list: dict[str, list[float]]  # ref_key -> std RMSE for each joint

    rmse_list: dict[str, list[list[float]]]  # ref_key -> tracked motion -> each joint RMSE


def load_rmse_data(
    tracked_trajectory: str,
    given_active_joints: list[int] | None,
    reference_prefix: str,
) -> RmseData:
    toml_path = Path(tracked_trajectory)
    with toml_path.open("rb") as f:
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

    # Load calculated RMSE
    if "rmse" not in config:
        msg = f"Not calculate RMSE yet: {toml_path}"
        raise RuntimeError(msg)

    all_rmse_mean_list = config["rmse"]["mean"]
    all_rmse_std_list = config["rmse"]["std"]
    rmse_mean_list = {}
    rmse_std_list = {}
    rmse_list = {}
    for key in reference_keys:
        rmse_mean_list[key] = config[key]["rmse"]["mean"]
        rmse_std_list[key] = config[key]["rmse"]["std"]
        rmse_list[key] = [x["rmse"] for x in config[key]["errors"]]

    return RmseData(
        toml_path,
        model_path,
        active_joints,
        reference_paths,
        reference_keys,
        motion_paths,
        all_rmse_mean_list,
        all_rmse_std_list,
        rmse_mean_list,
        rmse_std_list,
        rmse_list,
    )


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
    if output_dirpath.is_file():
        output_dirpath = output_dirpath.parent
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
        lines = ax.plot(t[mask], mean[mask], fmt, label=label, lw=3.0)
    else:
        mean, err1, err2 = calculate_mean_err(y, err_type=err_type)
        if err2 is None:
            eb = ax.errorbar(t[mask], mean[mask], yerr=err1[mask], capsize=capsize, fmt=fmt, label=label, lw=3.0)
        else:
            eb = ax.errorbar(
                t[mask], mean[mask], yerr=(err1[mask], err2[mask]), capsize=capsize, fmt=fmt, label=label, lw=3.0
            )
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
    joint_index: int,
    reference_data: Data,
    motion_data_list: list[list[Data]],  # MLP, PID, PID
    rmse_mean_list: list[float],  # MLP, PID, PID
    rmse_err_list: list[float],  # MLP, PID, PID
    tshift: float,
    tlim: tuple[float, float] | None,
    ref_label: str,
    labels: list[str] | None,
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
    title = f"Joint: {joint_index} | Reference: {reference_data.datapath.stem} | RMSE: "
    title += ",".join([f"{x:.2f}±{y:.2f}" for x, y in zip(rmse_mean_list, rmse_err_list, strict=True)])
    ylabel = "Joint position [0-100]"

    ax: Axes
    fig: Figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot reference
    t, y = load_timeseries(motion_data_list[0], f"q{joint_index}", tshift)
    t_ref = t if use_motion_reference else reference_data.t - tshift
    mask_ref = get_tlim_mask(t_ref, tlim)
    if use_motion_reference:
        y_ref = getattr(motion_data_list[0], f"qdes{joint_index}")
    else:
        y_ref = getattr(reference_data, f"q{joint_index}")
    (line,) = ax.plot(t_ref[mask_ref], y_ref[mask_ref], ls="--", label=ref_label, lw=3.0)

    # Plot tracked trajectories
    for i, motion_data_set in enumerate(motion_data_list):
        t, y = load_timeseries(motion_data_set, f"q{joint_index}", tshift)
        label = labels[i] if labels is not None else None
        line = plot_mean_err(ax, t, y, err_type, tlim, fmt="-", capsize=2, label=label)
        if fill:
            fill_between_err(ax, t, y, fill_err_type, tlim, line.get_color(), fill_alpha)

    ax.set_title(title, fontsize="xx-large")
    ax.set_ylim((-5, 105))
    ax.set_xlabel("time [s]", fontsize="xx-large")
    ax.set_ylabel(ylabel, fontsize="xx-large")
    ax.tick_params(axis="both", labelsize="xx-large")
    if show_legend:
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.0, fontsize="xx-large")
    fig.tight_layout()
    output_basename = f"{plot_prefix}_{joint_index:02d}_q"
    return save_figure(fig, output_dir_path, output_basename, ext, loaded_from=None, dpi=dpi)


def run(
    given_tracked_trajectory_paths: list[str],
    joints_str: str | None,
    output_dir_path: Path,
    reference_prefix: str,
    output_prefix: str,
    plot_prefix: str,
    plot_ext: list[str],
    tshift: float,
    tlim: tuple[float, float] | None,
    ref_label: str,
    labels: list[str] | None,
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
    all_saved_figures: list[Path] = []
    active_joints = resolve_joints_str(joints_str, dof=DEFAULT_DOF)
    event_logger().debug("Resolved active joints: %s", active_joints)

    # Load tracked trajectory motion file paths.
    rmse_data_set = [load_rmse_data(path, active_joints, reference_prefix) for path in given_tracked_trajectory_paths]
    event_logger().debug("Loaded tracked trajectory paths from: %s", given_tracked_trajectory_paths)

    # Plot reference and motion trajectories for all references.
    for reference_key, reference_path in zip(
        rmse_data_set[0].reference_keys,
        rmse_data_set[0].reference_paths,
        strict=True,
    ):
        reference_data = Data(reference_path)
        event_logger().debug("Loaded reference data: %s", reference_path)
        motion_data_list = [
            [Data(motion_path) for motion_path in rmse_data.motion_paths[reference_key]] for rmse_data in rmse_data_set
        ]  # MLP, PID, PID
        for i, joint_index in enumerate(active_joints):
            motion_plot_dir = output_dir_path / reference_key / output_prefix
            rmse_mean_list = [rmse_data.rmse_mean_list[reference_key][i] for rmse_data in rmse_data_set]
            rmse_err_list = [rmse_data.rmse_std_list[reference_key][i] for rmse_data in rmse_data_set]
            saved_figures = plot_all_motions(
                motion_plot_dir,
                plot_prefix,
                joint_index,
                reference_data,
                motion_data_list,
                rmse_mean_list,
                rmse_err_list,
                tshift,
                tlim,
                ref_label,
                labels,
                dpi=dpi,
                show_legend=show_legend,
                ext=plot_ext,
                use_motion_reference=use_motion_reference,
                err_type=err_type,
                fill=fill,
                fill_err_type=fill_err_type,
                fill_alpha=fill_alpha,
            )
            all_saved_figures.extend(saved_figures)
            if not show_screen:
                plt.close()

    if merge_plots:
        event_logger().debug("Merging the following figures: %s", all_saved_figures)
        merge_plot_figures(all_saved_figures, prefix="merged")

    if show_screen:
        plt.show()


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare reference and tracked trajectories.",
    )
    # Configuration
    parser.add_argument(
        "tracked_trajectories",
        nargs="+",
        help="List of paths to file 'tracked_trajectory.toml'.",
    )
    parser.add_argument(
        "-j",
        "--joints",
        nargs="*",
        required=True,
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
        default="compare_trajectory",
        help="Filename prefix that will be added to generated data files.",
    )
    parser.add_argument(
        "--plot-prefix",
        default="compare_trajectory",
        help="Filename prefix that will be added to plot figure.",
    )
    parser.add_argument("--tshift", type=float, default=0.0, help="time shift")
    parser.add_argument("--tlim", nargs="+", type=float, help="range of time")
    parser.add_argument("--ref-label", default="Reference", help="Label to display reference trajectory.")
    parser.add_argument("--labels", nargs="+", help="List of labels to display tracked trajectories.")
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
        output_dir = Path(args.tracked_trajectories[-1]).parent
    start_logging(sys.argv, output_dir, __name__, args.verbose)
    event_logger().info("Output directory: %s", output_dir)
    prepare_data_dir_path(output_dir, make_latest_symlink=False)
    event_logger().debug("Parsed arguments: %s", args)
    dpi = float(args.dpi) if args.dpi != "figure" else args.dpi

    # Start mainloop
    run(
        # configuration
        args.tracked_trajectories,
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
        args.ref_label,
        args.labels,
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
# jinx-local-words: "env pdf qdes rb rmse sd se tlim tshift usr vv"
# End:
