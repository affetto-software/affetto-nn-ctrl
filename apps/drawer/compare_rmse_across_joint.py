#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfWriter

from affetto_nn_ctrl.control_utility import resolve_joints_str
from affetto_nn_ctrl.data_handling import (
    prepare_data_dir_path,
)
from affetto_nn_ctrl.event_logging import event_logger, start_logging
from affetto_nn_ctrl.plot_utility import extract_common_parts, save_figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

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


def make_limit(limit: list[float] | tuple[float, ...] | None) -> tuple[float, float] | None:
    if limit is None or len(limit) == 0:
        return None
    if len(limit) == 1:
        return (-0.05, limit[0])
    return (min(limit), max(limit))


def _plot_rmse(
    ax: Axes,
    x: list[float] | np.ndarray,
    y: list[float],
    yerr: list[float],
    width: float,
    capsize: int,
    label: str | None,
    *,
    show_rmse: bool,
    show_line: bool,
) -> Axes:
    rects = ax.bar(x, y, yerr=yerr, width=width, capsize=capsize, label=label)
    if show_rmse:
        ax.bar_label(rects, label_type="center", fontsize="xx-large", fmt="%.3f")
    if show_line and len(rects.patches):
        c = rects.patches[0].get_facecolor()
        ax.plot(x, y, "--", color=c, lw=1.0)
    return ax


def plot_rmse(
    output_dir_path: Path,
    plot_prefix: str,
    active_joints: list[int],
    rmse_mean_list: list[list[float]],  # MLP, PID, PID
    rmse_err_list: list[list[float]],  # MLP, PID, PID
    reference_key: str | None,
    labels: list[str] | None,
    ylim: list[float] | tuple[float, float] | None,
    *,
    dpi: str | float,
    show_legend: bool,
    show_grid: Literal["both", "x", "y"],
    show_rmse: bool,
    ext: list[str],
) -> list[Path]:
    ylabel = "RMSE"
    if reference_key:
        ylabel += f" ({reference_key})"
    xlabel = "Joint index"
    xlabels = [f"#{x}" for x in active_joints]

    ax: Axes
    fig: Figure
    figsize = (3.5 * max(len(rmse_mean_list), len(active_joints)), 6)
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(active_joints))
    n = len(rmse_mean_list)
    width = 1.0 / (n + 1)

    y: list[float]
    yerr: list[float]
    for i, (rmse_mean, rmse_err) in enumerate(zip(rmse_mean_list, rmse_err_list, strict=True)):
        y = [rmse_mean[j] for j in range(len(active_joints))]
        yerr = [rmse_err[j] for j in range(len(active_joints))]
        label = labels[i] if labels is not None else None
        _plot_rmse(ax, x + i * width, y, yerr, width, 6, label, show_rmse=show_rmse, show_line=False)

    xticks_offset = 0.5 * (1.0 - 2.0 * width)
    ax.set_xticks(x + xticks_offset, xlabels, fontsize="xx-large")
    ax.minorticks_off()

    ylim = make_limit(ylim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, fontsize="xx-large")
    ax.set_ylabel(ylabel, fontsize="xx-large")
    ax.tick_params(axis="y", labelsize="xx-large")
    if show_grid in ("x", "y", "both"):
        ax.grid(axis=show_grid, visible=True)
    if show_legend:
        ax.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            ncols=len(rmse_mean_list),
            mode="expand",
            borderaxespad=0.0,
        )
    output_basename = f"{plot_prefix}"
    return save_figure(fig, output_dir_path, output_basename, ext, loaded_from=None, dpi=dpi)


def run(
    given_tracked_trajectory_paths: list[str],
    joints_str: str | None,
    output_dir_path: Path,
    reference_prefix: str,
    output_prefix: str,
    plot_prefix: str,
    plot_ext: list[str],
    labels: list[str] | None,
    ylim: list[float],
    *,
    merge_plots: bool,
    dpi: str | float,
    show_legend: bool,
    show_grid: Literal["both", "x", "y"],
    show_rmse: bool,
    show_screen: bool | None,
) -> None:
    # Resolve active joints.
    active_joints = resolve_joints_str(joints_str, dof=DEFAULT_DOF)
    event_logger().debug("Resolved active joints: %s", active_joints)

    # Load tracked trajectory motion file paths.
    rmse_data_set = [load_rmse_data(path, active_joints, reference_prefix) for path in given_tracked_trajectory_paths]
    event_logger().debug("Loaded tracked trajectory paths from: %s", given_tracked_trajectory_paths)

    all_saved_figures: list[Path] = []
    # Plot reference and motion trajectories for all references.
    for reference_key in rmse_data_set[0].reference_keys:
        bar_plot_dir = output_dir_path / reference_key / output_prefix
        rmse_mean_list = [rmse_data.rmse_mean_list[reference_key] for rmse_data in rmse_data_set]
        rmse_err_list = [rmse_data.rmse_std_list[reference_key] for rmse_data in rmse_data_set]
        saved_figures = plot_rmse(
            bar_plot_dir,
            plot_prefix,
            active_joints,
            rmse_mean_list,
            rmse_err_list,
            reference_key,
            labels,
            ylim,
            dpi=dpi,
            show_grid=show_grid,
            show_rmse=show_rmse,
            show_legend=show_legend,
            ext=plot_ext,
        )
        all_saved_figures.extend(saved_figures)
        if not show_screen:
            plt.close()

    bar_plot_dir = output_dir_path / output_prefix
    rmse_mean_list = [rmse_data.all_rmse_mean_list for rmse_data in rmse_data_set]
    rmse_err_list = [rmse_data.all_rmse_std_list for rmse_data in rmse_data_set]
    saved_figures = plot_rmse(
        bar_plot_dir,
        plot_prefix,
        active_joints,
        rmse_mean_list,
        rmse_err_list,
        None,
        labels,
        ylim,
        dpi=dpi,
        show_grid=show_grid,
        show_rmse=show_rmse,
        show_legend=show_legend,
        ext=plot_ext,
    )
    all_saved_figures.extend(saved_figures)

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
        default="compare_rmse",
        help="Filename prefix that will be added to generated data files.",
    )
    parser.add_argument(
        "--plot-prefix",
        default="compare_rmse",
        help="Filename prefix that will be added to plot figure.",
    )
    parser.add_argument("--labels", nargs="+", help="List of labels to display tracked trajectories.")
    parser.add_argument("--ylim", nargs="+", type=float, help="Limits along y-axis.")
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
        "--show-grid",
        default="none",
        help="which axis to show grid. choose from ['x','y','both', 'none'] (default: both)",
    )
    parser.add_argument(
        "--show-rmse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to show R2 score on each bar (default: True)",
    )
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
        args.labels,
        args.ylim,
        # boolean arguments
        merge_plots=args.merge_plots,
        dpi=dpi,
        show_legend=args.show_legend,
        show_grid=args.show_grid,
        show_rmse=args.show_rmse,
        show_screen=args.show_screen,
    )


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "env pdf rb rmse usr vv ylim"
# End:
