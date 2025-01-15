#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np

from affetto_nn_ctrl.data_handling import prepare_data_dir_path
from affetto_nn_ctrl.event_logging import FakeLogger, event_logger, get_logging_level_from_verbose_count, start_logging
from affetto_nn_ctrl.plot_utility import save_figure

if TYPE_CHECKING:
    from collections.abc import Generator

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.typing import ColorType

if sys.version_info < (3, 11):
    import tomli as tomllib  # type: ignore[reportMissingImports]
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
    *,
    show_arrows: bool,
) -> Axes:
    errbar = ax.errorbar(x, y, yerr=yerr, capsize=capsize, fmt=fmt, label=label)
    if show_arrows:
        _plot_arrow(ax, x, y, facecolor=errbar.lines[0].get_color())
    return ax


def _plot_arrow(ax: Axes, x: list[int], y: list[float], facecolor: ColorType) -> Axes:
    rng = np.random.default_rng()
    argmax = [i for i, _y in enumerate(y) if _y == max(y)]
    length = 1.0
    theta = np.pi * 0.03 + rng.uniform(-0.2, 0.2)
    offset: tuple[float, float] = (0.0, 0.0)
    for i in argmax:
        x_head = x[i] + offset[0]
        y_head = y[i] + offset[1]
        x_tail = x_head + length * np.cos(theta)
        y_tail = y_head + length * np.sin(theta)
        ax.annotate(
            "",
            xy=(x_head, y_head),
            xytext=(x_tail, y_tail),
            arrowprops={"facecolor": facecolor, "shrink": 0.0, "lw": 0.0},
        )
    return ax


def plot_scores(
    ax: Axes,
    collected_score_data: list[ScoreData],
    label: str | None,
    *,
    show_arrows: bool,
) -> Axes:
    steps = [data.steps for data in collected_score_data]
    scores = [data.score_mean for data in collected_score_data]
    errors = [data.score_std for data in collected_score_data]
    _plot_scores(ax, steps, scores, errors, fmt="--o", capsize=6, label=label, show_arrows=show_arrows)
    xticks = ax.get_xticks()
    if len(scores) > len(xticks):
        ax.set_xticks(steps)
        xlim = (min(steps) - 0.5, max(steps) + 0.5)
        ax.set_xlim(xlim)
    return ax


def make_plot_sets(
    basedir_list: list[str],
    adapter_list: list[str],
    regressor_list: list[str],
    scaler_list: list[str],
    dataset_tag_list: list[str],
) -> list[tuple[str, str, str, str, str]]:
    plot_sets = product(basedir_list, adapter_list, regressor_list, scaler_list, dataset_tag_list)
    return list(plot_sets)


def make_labels(
    basedir_list: list[str],
    adapter_list: list[str],
    regressor_list: list[str],
    scaler_list: list[str],
    dataset_tag_list: list[str],
    score_tag: str,
) -> list[str]:
    list_args = ([Path(b).name for b in basedir_list], adapter_list, scaler_list, regressor_list, dataset_tag_list)
    comparisons: list[list[str]] = [arg for arg in list_args if len(arg) > 1]
    labels = ["|".join(map(str, tpl)) for tpl in product(*comparisons)]
    if len(labels) == 1 and labels[0] == "":
        labels = [score_tag]
    return labels


def make_title(
    basedir_list: list[str],
    adapter_list: list[str],
    regressor_list: list[str],
    scaler_list: list[str],
    dataset_tag_list: list[str],
    default_title: str,
) -> str:
    list_args = ([Path(b).name for b in basedir_list], adapter_list, scaler_list, regressor_list, dataset_tag_list)
    consistents: list[str] = [arg[0] for arg in list_args if len(arg) == 1]
    if len(consistents) > 0:
        title = " | ".join(map(str, consistents))
    else:
        title = default_title
    return title


def make_output_prefix(
    basedir_list: list[str],
    adapter_list: list[str],
    regressor_list: list[str],
    scaler_list: list[str],
    dataset_tag_list: list[str],
    default_prefix: str,
) -> str:
    list_args = ([Path(b).name for b in basedir_list], adapter_list, scaler_list, regressor_list, dataset_tag_list)
    args_name = ("basedir", "adapter", "scaler", "regressor", "dataset")
    comparisons: list[str] = [name for arg, name in zip(list_args, args_name, strict=True) if len(arg) > 1]
    if len(comparisons) > 0:
        output_prefix = "compare_"
        output_prefix += "_".join(comparisons)
    else:
        output_prefix = default_prefix
    return output_prefix


def make_xlabel(adapter_list: list[str]) -> str:
    preview_bool = [a.startswith("preview") for a in adapter_list]
    delay_bool = [a.startswith("delay") for a in adapter_list]
    if any(preview_bool) and any(delay_bool):
        xlabel = "Preview/Delay steps"
    elif all(preview_bool):
        xlabel = "Preview steps"
    elif all(delay_bool):
        xlabel = "Delay steps"
    else:
        xlabel = "Steps"
    return xlabel


def plot_figure(
    basedir_list: list[str],
    adapter_list: list[str],
    regressor_list: list[str],
    scaler_list: list[str],
    dataset_tag_list: list[str],
    score_tag: str,
    filename: str,
    labels: list[str] | None,
    *,
    title: str | None,
    show_legend: bool,
    show_arrows: bool,
    show_grid: Literal["both", "x", "y"],
) -> tuple[Figure, Axes]:
    figsize = (8, 6)
    fig, ax = plt.subplots(figsize=figsize)
    list_args = [basedir_list, adapter_list, regressor_list, scaler_list, dataset_tag_list]
    if labels is None or len(labels) == 0:
        labels = make_labels(basedir_list, adapter_list, regressor_list, scaler_list, dataset_tag_list, score_tag)
    if title is None:
        title = make_title(basedir_list, adapter_list, regressor_list, scaler_list, dataset_tag_list, "scores")
    plot_sets = make_plot_sets(*list_args)
    if len(labels) != len(plot_sets):
        msg = "Inconsistent numbers of plot sets and labels: "
        f"{plot_sets}({len(plot_sets)}) vs {labels}({labels})"
        raise ValueError(msg)

    for (basedir, adapter, regressor, scaler, dataset_tag), label in zip(plot_sets, labels, strict=True):
        collected_score_data = collect_score_data(
            basedir,
            adapter,
            regressor,
            scaler,
            dataset_tag,
            score_tag,
            filename,
        )
        plot_scores(ax, collected_score_data, label, show_arrows=show_arrows)

    xlabel = make_xlabel(adapter_list)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Coefficient of determination")
    ax.set_ylim((-0.05, 1.05))
    if show_grid in ("x", "y", "both"):
        ax.grid(axis=show_grid, visible=True)
    if show_legend:
        ax.legend()
    ax.set_title(title)
    return fig, ax


def plot(
    basedir_list: list[str],
    adapter_list: list[str],
    regressor_list: list[str],
    scaler_list: list[str],
    dataset_tag_list: list[str],
    score_tag: str,
    filename: str,
    labels: list[str] | None,
    *,
    title: str | None,
    output_dir: Path,
    output_prefix: str | None,
    ext: list[str],
    dpi: float | str,
    show_legend: bool,
    show_arrows: bool,
    show_grid: Literal["both", "x", "y"],
    show_screen: bool,
) -> None:
    fig, _ = plot_figure(
        basedir_list,
        adapter_list,
        regressor_list,
        scaler_list,
        dataset_tag_list,
        score_tag,
        filename,
        labels,
        title=title,
        show_legend=show_legend,
        show_arrows=show_arrows,
        show_grid=show_grid,
    )
    output_prefix = make_output_prefix(
        basedir_list,
        adapter_list,
        regressor_list,
        scaler_list,
        dataset_tag_list,
        "scores",
    )
    save_figure(fig, output_dir, output_prefix, ext, dpi=dpi)
    if show_screen:
        plt.show()


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare scores across preview/delay steps")
    parser.add_argument("basedir", nargs="+", help="List of paths to directories containing scores data.")
    parser.add_argument("-a", "--adapter", nargs="+", help="Data adapter selector.")
    parser.add_argument("-s", "--scaler", nargs="+", help="Scaler selector.")
    parser.add_argument("-r", "--regressor", nargs="+", help="Regressor selector.")
    parser.add_argument("-d", "--dataset-tag", nargs="+", help="Dataset tag.")
    parser.add_argument("--score-tag", default="scores_000", help="Score data tag. (default: scores_000)")
    parser.add_argument("--score-filename", default="scores.toml", help="Scores filename. (default: scores.toml)")
    parser.add_argument("-l", "--labels", nargs="*", help="Label list shown in legend.")
    parser.add_argument("--title", help="figure title")
    parser.add_argument("-o", "--output-dir", help="Path to directory that figures are saved")
    parser.add_argument(
        "--output-prefix",
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
        default=True,
        help="whether to show legend (default: True)",
    )
    parser.add_argument(
        "--show-arrows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to show arrows at maximum scores (default: True)",
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
        args.labels,
        title=args.title,
        output_dir=output_dir,
        output_prefix=args.output_prefix,
        ext=args.ext,
        dpi=dpi,
        show_legend=args.show_legend,
        show_arrows=args.show_arrows,
        show_grid=args.show_grid,
        show_screen=args.show_screen,
    )


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "basedir dataset dir env facecolor lw rb regressor scaler usr vv"
# End:
