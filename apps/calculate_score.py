#!/usr/bin/env python

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from affetto_nn_ctrl import DEFAULT_SEED
from affetto_nn_ctrl.data_handling import (
    build_data_file_path,
    get_default_counter,
    prepare_data_dir_path,
    train_test_split_files,
)
from affetto_nn_ctrl.event_logging import event_logger, start_logging
from affetto_nn_ctrl.model_utility import (
    TrainedModel,
    load_datasets,
    load_train_datasets,
    load_trained_model,
)
from affetto_nn_ctrl.plot_utility import save_figure

DEFAULT_SHOW_SCREEN_NUM = 10


@dataclass
class CalculatedScore:
    test_dataset: Path
    plot_path: Path
    score: float


def save_scores(
    output_dir_path: Path,
    output_prefix: str,
    model_filepath: str,
    calculated_scores: list[CalculatedScore],
    ext: str = ".toml",
) -> None:
    if not ext.startswith("."):
        ext = f".{ext}"
    output = output_dir_path / f"{output_prefix}{ext}"

    arr = np.array([x.score for x in calculated_scores], dtype=float)
    text_lines = [
        "[model.performance]\n",
        f'model_path = "{model_filepath}"\n',
        f"score = {{ mean = {np.mean(arr)}, std = {np.std(arr)}, "
        f"argmax = {np.argmax(arr)}, argmin = {np.argmin(arr)} }}\n",
        "\n",
    ]
    for score in calculated_scores:
        text_lines.extend(
            [
                "[[model.performance.scores]]\n",
                f'test_dataset = "{score.test_dataset!s}"\n',
                f'plot_path = "{score.plot_path!s}"\n',
                f"score = {score.score}\n",
                "\n",
            ],
        )
    with output.open("w") as f:
        f.writelines(text_lines)
    event_logger().info("Calculated scores saved: %s", output)


def plot(
    output_dir_path: Path,
    plot_prefix: str,
    model: TrainedModel,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: CalculatedScore,
    *,
    dpi: str | float,
    show_legend: bool,
    ext: list[str],
) -> None:
    joints = model.adapter.params.active_joints
    dt = model.adapter.params.dt
    adapter_args = ", ".join([f"{k}={v}" for k, v in asdict(model.adapter.params).items() if k.endswith("step")])
    for i, joint_index in enumerate(joints):
        output_basename = f"{plot_prefix}_{joint_index}"
        title = (
            f"Joint: {joint_index} | "
            f"{model.adapter.__class__.__name__}({adapter_args}) | {model.model!s} | "
            f"{score.test_dataset.name} | Score: {score.score:.6f}"
        )
        cols = (i, i + len(joints))
        labels = ("ca", "cb")
        t_true = np.arange(len(y_true)) * dt
        t_pred = np.arange(len(y_pred)) * dt
        fig, ax = plt.subplots(figsize=(18, 6))
        for col, label in zip(cols, labels, strict=True):
            (line,) = ax.plot(t_true, y_true[:, col], ls="--", label=f"{label} (true)")
            ax.plot(t_pred, y_pred[:, col], c=line.get_color(), label=f"{label} (pred)")
        ax.set_title(title)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("pressure command [0-100]")
        if show_legend:
            ax.legend()
        save_figure(fig, output_dir_path, output_basename, ext, loaded_from=None, dpi=dpi)


def run(
    model_filepath: str,  # required
    dataset_paths: list[str],  # required
    glob_pattern: str,  # default: **/*.csv
    train_size: float | None,
    test_size: float | None,
    seed: int | None,
    output_dir_path: Path,
    output_prefix: str,
    plot_output_prefix: str,
    plot_prefix: str,
    plot_ext: list[str],
    *,
    shuffle: bool,
    split_in_each_directory: bool,
    overwrite: bool,
    dpi: str | float,
    show_legend: bool,
    show_screen: bool | None,
) -> None:
    # Load trained model.
    model = load_trained_model(model_filepath)
    event_logger().info("Trained model is loaded: %s", model_filepath)

    # Load test datasets to calculate the score.
    event_logger().debug("Loading datasets with following condition:")
    event_logger().debug("     Path list: %s", dataset_paths)
    event_logger().debug("  glob pattern: %s", glob_pattern)
    train_dataset_files, test_dataset_files = train_test_split_files(
        dataset_paths,
        test_size,
        train_size,
        glob_pattern,
        seed,
        shuffle=shuffle,
        split_in_each_directory=split_in_each_directory,
    )
    test_datasets = load_datasets(test_dataset_files)
    n_test_datasets = len(test_datasets)

    # Create output files counter.
    n_scores = 0
    if not overwrite:
        n_scores = len(list(output_dir_path.glob(f"{output_prefix}*")))
    cnt_scores = get_default_counter(n_scores)
    event_logger().debug("Calculated scores data counter initialized with %s", n_scores)
    scores_output_dir_path = build_data_file_path(output_dir_path, output_prefix, cnt_scores, ext="")

    # Create output plots counter.
    n_plots = 0
    if not overwrite:
        n_plots = len(list(scores_output_dir_path.glob(f"{plot_output_prefix}*")))
    cnt_plots = get_default_counter(n_plots)
    event_logger().debug("Performance plots counter initialized with %s", n_plots)

    # Calculate prediction and score of the trained model based on test datasets.
    calculated_scores: list[CalculatedScore] = []
    for i, test_dataset in enumerate(test_datasets):
        x_test, y_true = load_train_datasets(test_dataset, model.adapter)
        y_pred = model.predict(x_test)
        score = model.score(x_test, y_true)
        event_logger().info("[%s/%s] Calculated score: %s", i + 1, n_test_datasets, score)

        # Make plots and save them.
        plot_dir_path = build_data_file_path(scores_output_dir_path, plot_output_prefix, cnt_plots, ext="")
        calculated_score = CalculatedScore(test_dataset.datapath, plot_dir_path, score)
        plot(
            plot_dir_path,
            plot_prefix,
            model,
            y_true,
            y_pred,
            calculated_score,
            dpi=dpi,
            show_legend=show_legend,
            ext=plot_ext,
        )
        calculated_scores.append(calculated_score)

    save_scores(scores_output_dir_path, output_prefix, model_filepath, calculated_scores)

    if show_screen is None and n_test_datasets <= DEFAULT_SHOW_SCREEN_NUM:
        show_screen = True
    if show_screen:
        plt.show()


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate the score of the trained model based on test data sets.",
    )
    # Configuration
    # Input
    parser.add_argument(
        "model",
        help="Path to file in which trained model is encoded.",
    )
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        help="Path to file or directory containing data used to calculate score of trained model.",
    )
    parser.add_argument(
        "-g",
        "--glob-pattern",
        default="**/*.csv",
        help="Glob pattern to filter file to be loaded which is applied to each specified directory.",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        help="Ratio or number of files to use for training.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        help="Ratio or number of files to use for testing.",
    )
    parser.add_argument(
        "--seed",
        default=DEFAULT_SEED,
        type=int,
        help="Seed value given to random number generator.",
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Boolean. If True, shuffle files in dataset directory. (default: True)",
    )
    parser.add_argument(
        "--split-in-each-directory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Boolean. If True, splitting is done in each dataset directory. (default: False)",
    )
    # Parameters
    # Output
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory where calculated score and plot figure are stored. "
        "If nothing is provided, they are stored in the same directory with the trained model.",
    )
    parser.add_argument(
        "--output-prefix",
        default="scores",
        help="Filename prefix that will be added to calculated score.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Boolean. If True, restart counter of calculated performance data and overwrite existing files.",
    )
    parser.add_argument(
        "--plot-output-prefix",
        default="plots",
        help="Directory prefix that will be added to plot figures.",
    )
    parser.add_argument(
        "--plot-prefix",
        default="prediction",
        help="Filename prefix that will be added to plot figure.",
    )
    parser.add_argument(
        "-e",
        "--plot-ext",
        nargs="*",
        help="Filename prefix that will be added to plot figure.",
    )
    parser.add_argument("--dpi", default="figure", help="Figure DPI to be saved")
    parser.add_argument(
        "--show-legend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether show legend. (default: True)",
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
    if args.train_size is not None and args.train_size > 1:
        args.train_size = int(args.train_size)
    if args.test_size is not None and args.test_size > 1:
        args.test_size = int(args.test_size)

    # Prepare input/output
    if args.output is not None:
        output_dir = Path(args.output)
    else:
        output_dir = Path(args.model).parent
    start_logging(sys.argv, output_dir, __name__, args.verbose)
    event_logger().info("Output directory: %s", output_dir)
    prepare_data_dir_path(output_dir, make_latest_symlink=False)
    event_logger().debug("Parsed arguments: %s", args)
    dpi = float(args.dpi) if args.dpi != "figure" else args.dpi

    # Start mainloop
    run(
        # configuration
        # input
        args.model,
        args.datasets,
        args.glob_pattern,
        args.train_size,
        args.test_size,
        args.seed,
        # parameters
        # output
        output_dir,
        args.output_prefix,
        args.plot_output_prefix,
        args.plot_prefix,
        args.plot_ext,
        # boolean arguments
        shuffle=args.shuffle,
        split_in_each_directory=args.split_in_each_directory,
        overwrite=args.overwrite,
        dpi=dpi,
        show_legend=args.show_legend,
        show_screen=args.show_screen,
    )


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "argmax argmin cb csv dataset datasets env pdf pred regressor usr vv"
# End:
