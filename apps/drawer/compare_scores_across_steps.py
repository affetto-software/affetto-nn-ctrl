#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from affetto_nn_ctrl.data_handling import prepare_data_dir_path
from affetto_nn_ctrl.event_logging import FakeLogger, event_logger, get_logging_level_from_verbose_count, start_logging
from affetto_nn_ctrl.plot_utility import save_figure

if TYPE_CHECKING:
    from collections.abc import Generator

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib


@dataclass
class ScoreData:
    adapter_selector: str
    regressor_selector: str
    scaler_selector: str
    dataset_tag: str
    score_tag: str
    steps: int
    score_data_path: Path
    model_path: Path
    score_mean: float
    score_std: float
    test_datasets: list[Path]
    plot_paths: list[Path]
    scores: list[float]


def pop_path_name(fullpath: Path) -> Generator[str]:
    path = fullpath
    while str(path) != fullpath.root:
        last = path.name
        yield last
        path = path.parent


def load_score_data(score_data_path: Path) -> ScoreData:
    path_iter = pop_path_name(score_data_path)
    _ = next(path_iter)  # discard the first element
    score_tag = next(path_iter)
    dataset_tag = next(path_iter)
    scaler_selector = next(path_iter)
    regressor_selector = next(path_iter)
    adapter_selector = next(path_iter)
    steps = int(adapter_selector.split(".")[1][4:])
    with score_data_path.open("rb") as f:
        data = tomllib.load(f)
    performance_data = data["model"]["performance"]
    model_path = Path(performance_data["model_path"])
    score_mean = performance_data["score"]["mean"]
    score_std = performance_data["score"]["std"]
    test_datasets = [Path(x["test_dataset"]) for x in performance_data["scores"]]
    plot_paths = [Path(x["plot_path"]) for x in performance_data["scores"]]
    scores = [x["score"] for x in performance_data["scores"]]
    loaded_score_data = ScoreData(
        adapter_selector,
        regressor_selector,
        scaler_selector,
        dataset_tag,
        score_tag,
        steps,
        score_data_path,
        model_path,
        score_mean,
        score_std,
        test_datasets,
        plot_paths,
        scores,
    )
    event_logger().info("Score data loaded: %s", score_data_path)
    event_logger().debug("Loaded score data: %s", loaded_score_data)
    return loaded_score_data


def collect_score_data(
    basedir: str,
    adapter: str,
    regressor: str,
    scaler: str,
    dataset_tag: str,
    score_tag: str,
    filename: str,
) -> list[ScoreData]:
    basedir_path = Path(basedir)
    glob_pattern = f"{adapter}.step*/{regressor}/{scaler}/{dataset_tag}/{score_tag}/{filename}"
    collected_score_data_files = sorted(basedir_path.glob(glob_pattern))
    if len(collected_score_data_files) == 0:
        msg = f"No files found with {glob_pattern}: {basedir_path!s}"
        raise RuntimeError(msg)
    return [load_score_data(p) for p in collected_score_data_files]


def _plot_scores(
    ax: Axes,
    x: list[int],
    y: list[float],
    yerr: list[float],
    fmt: str,
    capsize: int,
    label: str | None,
) -> Axes:
    ax.errorbar(x, y, yerr=yerr, capsize=capsize, fmt=fmt, label=label)
    return ax


def plot_scores(ax: Axes, collected_score_data: list[ScoreData], label: str | None) -> Axes:
    steps = [data.steps for data in collected_score_data]
    scores = [data.score_mean for data in collected_score_data]
    errors = [data.score_std for data in collected_score_data]
    _plot_scores(ax, steps, scores, errors, fmt="--o", capsize=6, label=label)
    xticks = ax.get_xticks()
    if len(scores) > len(xticks):
        ax.set_xticks(steps)
        xlim = (min(steps) - 0.5, max(steps) + 0.5)
        ax.set_xlim(xlim)
    return ax


def plot_figure(
    basedir_list: list[str],
    adapter: str,
    regressor: str,
    scaler: str,
    dataset_tag: str,
    score_tag: str,
    filename: str,
    *,
    title: str | None,
    show_legend: bool,
    show_grid: str,
) -> tuple[Figure, Axes]:
    figsize = (8, 6)
    fig, ax = plt.subplots(figsize=figsize)
    label_list = ["not include desired velocity", "include desired velocity"]
    for basedir, label in zip(basedir_list, label_list, strict=False):
        collected_score_data = collect_score_data(
            basedir,
            adapter,
            regressor,
            scaler,
            dataset_tag,
            score_tag,
            filename,
        )
        plot_scores(ax, collected_score_data, label)

    if adapter.startswith("preview"):
        xlabel = "Preview steps"
    elif adapter.startswith("delay"):
        xlabel = "Delay steps"
    else:
        xlabel = "Steps"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Coefficient of determination")
    ax.set_ylim((-0.05, 1.05))
    if show_grid in ("x", "y", "both"):
        ax.grid(axis=show_grid, visible=True)
    if show_legend:
        ax.legend()
    if title:
        ax.set_title(title)
    return fig, ax


def make_title(
    adapter: str,
    regressor: str,
    scaler: str,
    dataset_tag: str,
) -> str:
    built_title = ""
    adapter_name = adapter.split(".")[0]
    built_title += f"{adapter_name} | {scaler} | {regressor} | {dataset_tag}"
    return built_title


def plot(
    basedir_list: list[str],
    adapter: str,
    regressor: str,
    scaler: str,
    dataset_tag: str,
    score_tag: str,
    filename: str,
    *,
    title: str | None,
    output_dir: Path,
    output_prefix: str,
    ext: list[str],
    dpi: float | str,
    show_legend: bool,
    show_grid: str,
    show_screen: bool,
) -> None:
    if title is None:
        title = make_title(adapter, regressor, scaler, dataset_tag)
    fig, _ = plot_figure(
        basedir_list,
        adapter,
        regressor,
        scaler,
        dataset_tag,
        score_tag,
        filename,
        title=title,
        show_legend=show_legend,
        show_grid=show_grid,
    )
    save_figure(fig, output_dir, output_prefix, ext, dpi=dpi)
    if show_screen:
        plt.show()


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare scores across preview/delay steps")
    parser.add_argument("basedir", nargs="+", help="List of paths to directories containing scores data.")
    parser.add_argument("-a", "--adapter", required=True, help="Data adapter selector.")
    parser.add_argument("-s", "--scaler", required=True, help="Scaler selector.")
    parser.add_argument("-r", "--regressor", required=True, help="Regressor selector.")
    parser.add_argument("-d", "--dataset-tag", required=True, help="Dataset tag.")
    parser.add_argument("--score-tag", default="scores_000", help="Score data tag. (default: scores_000)")
    parser.add_argument("--score-filename", default="scores.toml", help="Scores filename. (default: scores.toml)")
    parser.add_argument("--title", help="figure title")
    parser.add_argument("-o", "--output-dir", help="Path to directory that figures are saved")
    parser.add_argument(
        "--output-prefix",
        default="compare_steps",
        help="Filename prefix that will be added to plot figure.",
    )
    parser.add_argument(
        "-e",
        "--save-ext",
        dest="ext",
        nargs="*",
        help="extensions that the figure will be saved as",
    )
    parser.add_argument("--dpi", default="figure", help="figure DPI to be saved")
    parser.add_argument(
        "--show-screen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether plot screen is shown or not (default: True)",
    )
    parser.add_argument(
        "--show-legend",
        action=argparse.BooleanOptionalAction,
        help="whether to show legend (default: True)",
    )
    parser.add_argument(
        "--show-grid",
        default="both",
        help="which axis to show grid. choose from ['x','y','both', 'none'] (default: both)",
    )
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
    if args.verbose > 0 and isinstance(event_logger(), FakeLogger):
        event_logger().setLevel(get_logging_level_from_verbose_count(args.verbose))

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.basedir[0])
    start_logging(sys.argv, output_dir, __name__, args.verbose)
    event_logger().info("Output directory: %s", output_dir)
    prepare_data_dir_path(output_dir, make_latest_symlink=False)
    event_logger().debug("Parsed arguments: %s", args)
    dpi = float(args.dpi) if args.dpi != "figure" else args.dpi

    plot(
        args.basedir,
        args.adapter,
        args.regressor,
        args.scaler,
        args.dataset_tag,
        args.score_tag,
        args.score_filename,
        title=args.title,
        output_dir=output_dir,
        output_prefix=args.output_prefix,
        ext=args.ext,
        dpi=dpi,
        show_legend=args.show_legend,
        show_grid=args.show_grid,
        show_screen=args.show_screen,
    )


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "basedir dataset dir env rb regressor scaler usr vv"
# End:
