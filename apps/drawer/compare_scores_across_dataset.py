#!/usr/bin/env python

from __future__ import annotations

import argparse
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
    from matplotlib.figure import Figure

if sys.version_info < (3, 11):
    import tomli as tomllib  # type: ignore[reportMissingImports]
else:
    import tomllib

plt.style.use(["science", "notebook", "grid"])
plt.rcParams.update({"axes.grid.axis": "y"})


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


# dataset_tag -> adapter -> score_data
CollectedScoreData: TypeAlias = dict[str, dict[str, ScoreData]]


def collect_score_data(
    basedir_list: list[str],
    step: int,
    adapter_list: list[str],
    regressor: str,
    scaler: str,
    dataset_tag_list: list[str],
    score_tag: str,
    filename: str,
) -> CollectedScoreData:
    collected_score_data: CollectedScoreData = {}
    for dataset_tag, adapter in product(dataset_tag_list, adapter_list):
        if adapter == "without-adapter":
            _adapter = "delay-states-all"
            _step = 0
        else:
            _adapter = adapter
            _step = step
        score_data_file = f"{_adapter}.step{_step:02d}/{regressor}/{scaler}/{dataset_tag}/{score_tag}/{filename}"
        found = False
        for basedir in basedir_list:
            score_data_path = Path(basedir) / score_data_file
            if score_data_path.exists() and not found:
                score_data = load_score_data(score_data_path)
                collected_score_data.setdefault(dataset_tag, {})[adapter] = score_data
                found = True
            elif found:
                msg = f"Duplicate score data found: {score_data_path}"
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
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
    "mlp.default-iter800": "MLP(#100)/ReLU",
    "mlp.layer200-iter800": "MLP(#200)/ReLU",
    "mlp.layer100-100-iter800": "MLP(#100-#100)/ReLU",
    "mlp.layer100-100-iter800-identity": "MLP(#100-#100)/Identity",
    "mlp.layer100-100-iter800-logistic": "MLP(#100-#100)/Logistic",
    "mlp.layer100-100-iter800-logistic-lbfgs": "MLP(#100-#100)/Logistic/L-FBFGS",
    "mlp.layer100-100-iter800-logistic-sgd": "MLP(#100-#100)/Logistic/SGD",
    "mlp.layer100-100-iter800-tanh": "MLP(#100-#100)/tanh",
    "mlp.layer100-100-iter800-tanh-lbfgs": "MLP(#100-#100)/tanh/L-FBFGS",
    "mlp.layer100-100-iter800-tanh-sgd": "MLP(#100-#100)/tanh/SGD",
}
dataset_tag_names = {
    "step_sync_slow": "Discont/Sync/Slow",
    "step_sync_middle": "Discont/Sync/Middle",
    "step_sync_fast": "Discont/Sync/Fast",
    "step_sync_all": "Discont/Sync/All",
    "trapez_sync_slow": "Cont/Sync/Slow",
    "trapez_sync_middle": "Cont/Sync/Middle",
    "trapez_sync_fast": "Cont/Sync/Fast",
    "trapez_sync_all": "Cont/Sync/All",
    "mix_sync_slow": "Mix/Sync/Slow",
    "mix_sync_middle": "Mix/Sync/Middle",
    "mix_sync_fast": "Mix/Sync/Fast",
    "mix_sync_all": "Mix/Sync/All",
    "step_async_slow": "Discont/Async/Slow",
    "step_async_middle": "Discont/Async/Middle",
    "step_async_fast": "Discont/Async/Fast",
    "step_async_all": "Discont/Async/All",
    "trapez_async_slow": "Cont/Async/Slow",
    "trapez_async_middle": "Cont/Async/Middle",
    "trapez_async_fast": "Cont/Async/Fast",
    "trapez_async_all": "Cont/Async/All",
    "mix_async_slow": "Mix/Async/Slow",
    "mix_async_middle": "Mix/Async/Middle",
    "mix_async_fast": "Mix/Async/Fast",
    "mix_async_all": "Mix/Async/All",
    "step_mix_slow": "Discont/Mix/Slow",
    "step_mix_middle": "Discont/Mix/Middle",
    "step_mix_fast": "Discont/Mix/Fast",
    "step_mix_all": "Discont/Mix/All",
    "trapez_mix_slow": "Cont/Mix/Slow",
    "trapez_mix_middle": "Cont/Mix/Middle",
    "trapez_mix_fast": "Cont/Mix/Fast",
    "trapez_mix_all": "Cont/Mix/All",
    "mix_mix_slow": "Mix/Mix/Slow",
    "mix_mix_middle": "Mix/Mix/Middle",
    "mix_mix_fast": "Mix/Mix/Fast",
    "mix_mix_all": "Mix/Mix/All",
}


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
        ax.bar_label(rects, label_type="center", fontsize="xx-large", fmt="%.3f")
    if show_line and len(rects.patches):
        c = rects.patches[0].get_facecolor()
        ax.plot(x, y, "--", color=c, lw=1.0)
    return ax


def plot_scores(
    ax: Axes,
    collected_score_data: CollectedScoreData,
    adapter_list: list[str],
    dataset_tag_list: list[str],
    labels: list[str],
    xlabels: list[str],
    *,
    show_r2: bool,
    show_lines: list[str],
) -> Axes:
    x = np.arange(len(dataset_tag_list))
    n = len(adapter_list)
    width = 1.0 / (n + 1)

    y: list[float]
    yerr: list[float]
    for i, (adapter, label) in enumerate(zip(adapter_list, labels, strict=True)):
        y = [collected_score_data[tag][adapter].score_mean for tag in dataset_tag_list]
        yerr = [collected_score_data[tag][adapter].score_std for tag in dataset_tag_list]
        _plot_scores(ax, x + i * width, y, yerr, width, 6, label, show_r2=show_r2, show_line=adapter in show_lines)

    xticks_offset = 0.5 * (1.0 - 2.0 * width)
    ax.set_xticks(x + xticks_offset, xlabels, fontsize="xx-large")
    ax.minorticks_off()
    return ax


def make_labels(
    basedir_list: list[str],
    step: int,
    adapter_list: list[str],
    regressor_list: list[str],
    scaler_list: list[str],
    dataset_tag_list: list[str],
    score_tag: str,
) -> list[str]:
    _ = basedir_list, step, regressor_list, scaler_list, dataset_tag_list, score_tag
    return [adapter_names.get(x, x) for x in adapter_list]


def make_xlabels(
    basedir_list: list[str],
    step: int,
    adapter_list: list[str],
    regressor_list: list[str],
    scaler_list: list[str],
    dataset_tag_list: list[str],
    score_tag: str,
) -> list[str]:
    _ = basedir_list, step, adapter_list, regressor_list, scaler_list, score_tag
    return [dataset_tag_names.get(x, x) for x in dataset_tag_list]


def make_title(
    basedir_list: list[str],
    step: int,
    adapter_list: list[str],
    regressor_list: list[str],
    scaler_list: list[str],
    dataset_tag_list: list[str],
    default_title: str,
) -> str:
    _ = basedir_list, adapter_list, dataset_tag_list
    if len(regressor_list) > 0 and len(scaler_list) > 0:
        regressor = regressor_list[0]
        scaler = scaler_list[0]
        title = f"{regressor_names.get(regressor, regressor)} ({scaler_names.get(scaler, scaler)}), step={step}"
    else:
        title = default_title
    return title


def make_output_prefix(
    basedir_list: list[str],
    step: int,
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
    return f"{output_prefix}_step{step:02d}"


def make_limit(limit: list[float] | tuple[float, ...] | None) -> tuple[float, float] | None:
    if limit is None or len(limit) == 0:
        return None
    if len(limit) == 1:
        return (-0.05, limit[0])
    return (min(limit), max(limit))


def plot_figure(
    basedir_list: list[str],
    step: int,
    adapter_list: list[str],
    regressor_list: list[str],
    scaler_list: list[str],
    dataset_tag_list: list[str],
    score_tag: str,
    filename: str,
    *,
    title: str | None,
    labels: list[str] | None,
    xlabels: list[str] | None,
    ylim: list[float] | tuple[float, float] | None,
    show_legend: bool,
    show_grid: Literal["both", "x", "y"],
    show_r2: bool,
    show_lines: list[str] | None,
) -> tuple[Figure, Axes]:
    figsize = (3.5 * max(len(dataset_tag_list), len(adapter_list)), 6)
    fig, ax = plt.subplots(figsize=figsize)
    if labels is None or len(labels) == 0:
        labels = make_labels(
            basedir_list,
            step,
            adapter_list,
            regressor_list,
            scaler_list,
            dataset_tag_list,
            score_tag,
        )
    if xlabels is None or len(xlabels) == 0:
        xlabels = make_xlabels(
            basedir_list,
            step,
            adapter_list,
            regressor_list,
            scaler_list,
            dataset_tag_list,
            score_tag,
        )
    if title is not None and title.lower() == "default":
        title = make_title(
            basedir_list,
            step,
            adapter_list,
            regressor_list,
            scaler_list,
            dataset_tag_list,
            "scores",
        )
    if show_lines is None:
        show_lines = []
    ylim = make_limit(ylim)
    collected_score_data = collect_score_data(
        basedir_list,
        step,
        adapter_list,
        regressor_list[0],
        scaler_list[0],
        dataset_tag_list,
        score_tag,
        filename,
    )
    plot_scores(
        ax,
        collected_score_data,
        adapter_list,
        dataset_tag_list,
        labels,
        xlabels,
        show_r2=show_r2,
        show_lines=show_lines,
    )

    ax.set_ylabel(r"Coefficient of determination, $R^2$", fontsize="xx-large")
    ax.tick_params(axis="y", labelsize="xx-large")
    ax.set_ylim(ylim)
    if show_grid in ("x", "y", "both"):
        ax.grid(axis=show_grid, visible=True)
    if show_legend:
        ax.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            ncols=len(adapter_list),
            mode="expand",
            borderaxespad=0.0,
        )
    if title:
        fig.suptitle(title)
    return fig, ax


def plot(
    basedir_list: list[str],
    step: int,
    adapter_list: list[str],
    regressor_list: list[str],
    scaler_list: list[str],
    dataset_tag_list: list[str],
    score_tag: str,
    filename: str,
    *,
    title: str | None,
    labels: list[str] | None,
    xlabels: list[str] | None,
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
    fig, _ = plot_figure(
        basedir_list,
        step,
        adapter_list,
        regressor_list,
        scaler_list,
        dataset_tag_list,
        score_tag,
        filename,
        title=title,
        labels=labels,
        xlabels=xlabels,
        ylim=ylim,
        show_legend=show_legend,
        show_grid=show_grid,
        show_r2=show_r2,
        show_lines=show_lines,
    )
    if output_prefix is None:
        output_prefix = make_output_prefix(
            basedir_list,
            step,
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
    parser = argparse.ArgumentParser(description="Generate LaTeX table to compare scores across regressor models")
    parser.add_argument("basedir", nargs="+", help="List of paths to directories containing scores data.")
    parser.add_argument("-i", "--step", type=int, default=9, help="Delay/Preview step to show in table.")
    parser.add_argument("-a", "--adapter", nargs="+", help="Data adapter selector.")
    parser.add_argument("-s", "--scaler", nargs="+", help="Scaler selector.")
    parser.add_argument("-r", "--regressor", nargs="+", help="Regressor selector.")
    parser.add_argument("-d", "--dataset-tag", nargs="+", help="Dataset tag.")
    parser.add_argument("--score-tag", default="scores_000", help="Score data tag. (default: scores_000)")
    parser.add_argument("--score-filename", default="scores.toml", help="Scores filename. (default: scores.toml)")
    parser.add_argument("--title", help="figure title")
    parser.add_argument("--labels", nargs="*", help="Bar label list. (Adapter list)")
    parser.add_argument("--xlabels", nargs="*", help="X-axis label list. (Dataset tag list)")
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
        args.adapter,
        args.regressor,
        args.scaler,
        args.dataset_tag,
        args.score_tag,
        args.score_filename,
        title=args.title,
        labels=args.labels,
        xlabels=args.xlabels,
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
# jinx-local-words: "Discont ReLU async basedir dataset dir env lbfgs maxabs minmax mlp rb regressor scaler sgd tanh trapez usr vv xlabels ylim" # noqa: E501
# End:
