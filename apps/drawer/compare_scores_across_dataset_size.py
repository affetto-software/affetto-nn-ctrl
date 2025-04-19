#!/usr/bin/env python
# ruff: noqa: C901, PLR0912

from __future__ import annotations

import argparse
import re
import sys
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np

from affetto_nn_ctrl.data_handling import prepare_data_dir_path
from affetto_nn_ctrl.event_logging import FakeLogger, event_logger, get_logging_level_from_verbose_count, start_logging
from affetto_nn_ctrl.plot_utility import save_figure

if TYPE_CHECKING:
    from collections.abc import Generator

    from matplotlib.axes import Axes

if sys.version_info < (3, 11):
    import tomli as tomllib  # type: ignore[reportMissingImports]
else:
    import tomllib

plt.style.use(["science", "notebook", "grid"])
plt.rcParams.update({"axes.grid.axis": "y"})


@dataclass
class ScoreData:
    regressor_selector: str
    adapter_selector: str
    dataset_size: str
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
    dataset_size = next(path_iter)
    adapter_selector = next(path_iter)
    steps = int(adapter_selector.split(".")[1][4:])
    regressor_selector = next(path_iter)
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
        regressor_selector,
        adapter_selector,
        dataset_size,
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


# regressor -> adapter -> dataset_size -> score_data
CollectedScoreData: TypeAlias = dict[str, dict[str, dict[str, ScoreData]]]


def collect_score_data(
    basedir_list: list[str],
    step: int,
    regressor_list: list[str],
    adapter_list: list[str],
    dataset_size_list: list[str],
    score_tag: str,
    filename: str,
) -> CollectedScoreData:
    collected_score_data: CollectedScoreData = {}
    for regressor, adapter, dataset_size in product(regressor_list, adapter_list, dataset_size_list):
        if adapter == "without-adapter":
            _adapter = "delay-states-all"
            _step = 0
        else:
            _adapter = adapter
            _step = step
        score_data_file = f"{regressor}/{_adapter}.step{_step:02d}/{dataset_size}/{score_tag}/{filename}"
        found = False
        for basedir in basedir_list:
            score_data_path = Path(basedir) / score_data_file
            if score_data_path.exists():
                score_data = load_score_data(score_data_path)
                collected_score_data.setdefault(regressor, {}).setdefault(adapter, {})[dataset_size] = score_data
                if found:
                    msg = f"Duplicate score data found: overwritten by {score_data_path}"
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)
                found = True
        if not found:
            msg = f"No score data found: {score_data_file} in {basedir_list}"
            raise RuntimeError(msg)
    return collected_score_data


adapter_names = {
    "delay-states": "States delay",
    "delay-states-all": "Recursive states delay",
    "preview-ref": "Reference preview",
    "without-adapter": "W/o delay embedding",
}
scaler_names = {
    "none": "W/o scaler",
    "minmax": "MinMax",
    "maxabs": "MaxAbs",
    "std": "Std",
    "robust": "Robust",
}
regressor_names = {
    "linear.default": "Linear",
    "ridge.default": "Ridge",
    "mlp.best": "MLP",
    "esn.best": "ESN",
}


def dataset_size_as_string(dataset_size: str) -> str:
    if not dataset_size:
        return ""

    match = re.search(r"\d+", dataset_size)
    if match:
        return str(int(match.group()))
    return ""


def _plot_scores(
    ax: Axes,
    x: list[float] | np.ndarray,
    y: list[float],
    yerr: list[float],
    width: float,
    capsize: int,
    label: str | None,
    *,
    show_r2: bool,
    show_line: bool,
) -> Axes:
    rects = ax.bar(x, y, yerr=yerr, width=width, capsize=capsize, label=label)
    if show_r2:
        ax.bar_label(rects, label_type="center")
    if show_line and len(rects.patches):
        c = rects.patches[0].get_facecolor()
        ax.plot(x, y, "--", color=c, lw=1.0)
    return ax


def plot_scores_compare_regressor(
    ax: Axes,
    collected_score_data: CollectedScoreData,
    regressor_list: list[str],
    adapter: str,
    dataset_size_list: list[str],
    *,
    show_r2: bool,
    show_lines: list[str],
    labels: list[str] | None = None,
    xlabels: list[str] | None = None,
    title: str | None = None,
) -> Axes:
    if labels is None:
        labels = [regressor_names.get(x, x) for x in regressor_list]
    if xlabels is None:
        xlabels = [dataset_size_as_string(x) for x in dataset_size_list]

    x = np.arange(len(dataset_size_list))
    n = len(regressor_list)
    width = 1.0 / (n + 1)

    y: list[float]
    yerr: list[float]
    for i, (regressor, label) in enumerate(zip(regressor_list, labels, strict=True)):
        y = [collected_score_data[regressor][adapter][size].score_mean for size in dataset_size_list]
        yerr = [collected_score_data[regressor][adapter][size].score_std for size in dataset_size_list]
        _plot_scores(ax, x + i * width, y, yerr, width, 6, label, show_r2=show_r2, show_line=adapter in show_lines)

    xticks_offset = 0.5 * (1.0 - 2.0 * width)
    ax.set_xticks(x + xticks_offset, xlabels)
    ax.minorticks_off()

    if title is not None:
        ax.set_title(title)
    return ax


def plot_scores_compare_adapter(
    ax: Axes,
    collected_score_data: CollectedScoreData,
    regressor: str,
    adapter_list: list[str],
    dataset_size_list: list[str],
    *,
    show_r2: bool,
    show_lines: list[str],
    labels: list[str] | None = None,
    xlabels: list[str] | None = None,
    title: str | None = None,
) -> Axes:
    if labels is None:
        labels = [adapter_names.get(x, x) for x in adapter_list]
    if xlabels is None:
        xlabels = [dataset_size_as_string(x) for x in dataset_size_list]

    x = np.arange(len(dataset_size_list))
    n = len(adapter_list)
    width = 1.0 / (n + 1)

    y: list[float]
    yerr: list[float]
    for i, (adapter, label) in enumerate(zip(adapter_list, labels, strict=True)):
        y = [collected_score_data[regressor][adapter][size].score_mean for size in dataset_size_list]
        yerr = [collected_score_data[regressor][adapter][size].score_std for size in dataset_size_list]
        _plot_scores(ax, x + i * width, y, yerr, width, 6, label, show_r2=show_r2, show_line=adapter in show_lines)

    xticks_offset = 0.5 * (1.0 - 2.0 * width)
    ax.set_xticks(x + xticks_offset, xlabels)
    ax.minorticks_off()

    if title is not None:
        ax.set_title(title)
    return ax


def make_limit(limit: list[float] | tuple[float, ...] | None) -> tuple[float, float] | None:
    if limit is None or len(limit) == 0:
        return (0.0, 1.2)
    if len(limit) == 1:
        return (0.0, limit[0])
    return (min(limit), max(limit))


def plot_figure(
    basedir_list: list[str],
    step: int,
    regressor_list: list[str],
    adapter_list: list[str],
    dataset_size_list: list[str],
    score_tag: str,
    filename: str,
    *,
    given_title: str | None,
    ylim: list[float] | tuple[float, float] | None,
    output_dir: Path,
    given_output_prefix: str | None,
    ext: list[str],
    dpi: float | str,
    show_legend: bool,
    show_grid: Literal["both", "x", "y"],
    show_r2: bool,
    show_lines: list[str] | None,
) -> None:
    collected_score_data = collect_score_data(
        basedir_list,
        step,
        regressor_list,
        adapter_list,
        dataset_size_list,
        score_tag,
        filename,
    )
    for i, adapter in enumerate(adapter_list):
        figsize = (3.5 * max(len(dataset_size_list), len(regressor_list)), 6)
        fig, ax = plt.subplots(figsize=figsize)
        if given_title is not None and given_title.lower() == "default":
            title = adapter_names.get(adapter, adapter)
        else:
            title = given_title
        if show_lines is None:
            show_lines = []
        ylim = make_limit(ylim)
        plot_scores_compare_regressor(
            ax,
            collected_score_data,
            regressor_list,
            adapter,
            dataset_size_list,
            show_r2=show_r2,
            show_lines=show_lines,
            title=title,
        )

        ax.set_ylabel(r"Coefficient of determination, $R^2$")
        ax.set_ylim(ylim)
        if show_grid in ("x", "y", "both"):
            ax.grid(axis=show_grid, visible=True)
        if show_legend:
            ax.legend(
                bbox_to_anchor=(0.0, 0.88, 1.0, 0.102),
                loc="lower left",
                ncols=len(regressor_list),
                mode="expand",
                borderaxespad=0.0,
            )
        if given_output_prefix is None:
            output_prefix = "compare_dataset_size_regressor"
        else:
            output_prefix = given_output_prefix
        save_figure(fig, output_dir, f"{output_prefix}_{i:02d}", ext, dpi=dpi)

    for i, regressor in enumerate(regressor_list):
        figsize = (3.5 * max(len(dataset_size_list), len(regressor_list)), 6)
        fig, ax = plt.subplots(figsize=figsize)
        if given_title is not None and given_title.lower() == "default":
            title = regressor_names.get(regressor, regressor)
        else:
            title = given_title
        if show_lines is None:
            show_lines = []
        ylim = make_limit(ylim)
        plot_scores_compare_adapter(
            ax,
            collected_score_data,
            regressor,
            adapter_list,
            dataset_size_list,
            show_r2=show_r2,
            show_lines=show_lines,
            title=title,
        )

        ax.set_xlabel(r"Dataset size")
        ax.set_ylabel(r"Coefficient of determination, $R^2$")
        ax.set_ylim(ylim)
        if show_grid in ("x", "y", "both"):
            ax.grid(axis=show_grid, visible=True)
        if show_legend:
            ax.legend(
                bbox_to_anchor=(0.0, 0.88, 1.0, 0.102),
                loc="lower left",
                ncols=len(regressor_list),
                mode="expand",
                borderaxespad=0.0,
            )
        if given_output_prefix is None:
            output_prefix = "compare_dataset_size_adapter"
        else:
            output_prefix = given_output_prefix
        save_figure(fig, output_dir, f"{output_prefix}_{i:02d}", ext, dpi=dpi)


def plot(
    basedir_list: list[str],
    step: int,
    regressor_list: list[str],
    adapter_list: list[str],
    dataset_size_list: list[str],
    score_tag: str,
    filename: str,
    *,
    title: str | None,
    ylim: list[float] | None,
    output_dir: Path,
    output_prefix: str | None,
    ext: list[str],
    dpi: float | str,
    show_legend: bool,
    show_grid: Literal["both", "x", "y"],
    show_screen: bool,
    show_r2: bool,
    show_lines: list[str] | None,
) -> None:
    plot_figure(
        basedir_list,
        step,
        regressor_list,
        adapter_list,
        dataset_size_list,
        score_tag,
        filename,
        given_title=title,
        ylim=ylim,
        output_dir=output_dir,
        given_output_prefix=output_prefix,
        ext=ext,
        dpi=dpi,
        show_legend=show_legend,
        show_grid=show_grid,
        show_r2=show_r2,
        show_lines=show_lines,
    )
    if show_screen:
        plt.show()


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LaTeX table to compare scores across regressor models")
    parser.add_argument("basedir", nargs="+", help="List of paths to directories containing scores data.")
    parser.add_argument("-i", "--step", type=int, default=9, help="Delay/Preview step to show in table.")
    parser.add_argument("-r", "--regressor", nargs="+", help="Regressor selector.")
    parser.add_argument("-a", "--adapter", nargs="+", help="Data adapter selector.")
    parser.add_argument("-d", "--dataset-size", nargs="+", help="Dataset tag.")
    parser.add_argument("--score-tag", default="scores_000", help="Score data tag. (default: scores_000)")
    parser.add_argument("--score-filename", default="scores.toml", help="Scores filename. (default: scores.toml)")
    parser.add_argument("--title", default="default", help="figure title")
    parser.add_argument("--ylim", nargs="+", type=float, help="Limits along y-axis.")
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
        "--show-grid",
        default="none",
        help="which axis to show grid. choose from ['x','y','both', 'none'] (default: both)",
    )
    parser.add_argument(
        "--show-r2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to show R2 score on each bar (default: True)",
    )
    parser.add_argument(
        "--show-lines",
        nargs="+",
        default=["preview-ref"],
        help="List of data adapter names to show line plots.",
    )
    parser.add_argument(
        "--no-show-lines",
        action="store_true",
        help="Never show line plots.",
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
    if args.no_show_lines:
        args.show_lines = []

    plot(
        args.basedir,
        args.step,
        args.regressor,
        args.adapter,
        args.dataset_size,
        args.score_tag,
        args.score_filename,
        title=args.title,
        ylim=args.ylim,
        output_dir=output_dir,
        output_prefix=args.output_prefix,
        ext=args.ext,
        dpi=dpi,
        show_legend=args.show_legend,
        show_grid=args.show_grid,
        show_screen=args.show_screen,
        show_r2=args.show_r2,
        show_lines=args.show_lines,
    )


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "Discont ReLU async basedir dataset dir env esn lbfgs maxabs minmax mlp noqa rb regressor scaler sgd tanh trapez usr vv xlabels ylim" # noqa: E501
# End:
