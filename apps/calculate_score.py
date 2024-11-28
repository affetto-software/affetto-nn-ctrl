#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from affetto_nn_ctrl.data_handling import (
    prepare_data_dir_path,
)
from affetto_nn_ctrl.event_logging import event_logger, start_logging
from affetto_nn_ctrl.model_utility import (
    DataAdapterBase,
    load_datasets,
    load_train_datasets,
    load_trained_model,
)
from affetto_nn_ctrl.plot_utility import save_figure

if TYPE_CHECKING:
    import numpy as np


def save_score(output_dir_path: Path, output_prefix: str, score: float, ext: str = ".toml") -> None:
    if not ext.startswith("."):
        ext = f".{ext}"
    output = output_dir_path / f"{output_prefix}{ext}"

    text = f"""\
[model.result]
score = {score}
"""
    with output.open("w") as f:
        f.write(text)
    event_logger().info("Score saved: %s", output)


def plot(
    output_dir_path: Path,
    plot_prefix: str,
    adapter: DataAdapterBase,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    dpi: str | float,
    show_legend: bool,
    ext: list[str],
) -> None:
    joints = adapter.params.active_joints
    for i, joint_index in enumerate(joints):
        output_basename = f"{plot_prefix}_{joint_index}"
        title = f"Command data and prediction (joint: {joint_index})"
        cols = (i, i + len(joints))
        labels = ("ca", "cb")
        fig, ax = plt.subplots(figsize=(12, 6))
        for col, label in zip(cols, labels, strict=True):
            (line,) = ax.plot(y_true[:, col], ls="--", label=f"{label} (true)")
            ax.plot(y_pred[:, col], c=line.get_color(), label=f"{label} (pred)")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if show_legend:
            ax.legend()
        save_figure(fig, output_dir_path, output_basename, ext, dpi=dpi)


def run(
    model_filepath: str,  # required
    dataset_paths: list[str],  # required
    glob_pattern: str,  # default: **/*.csv
    n_pickup: int | None,
    output_dir_path: Path,
    output_prefix: str,
    plot_prefix: str,
    plot_ext: list[str],
    *,
    dpi: str | float,
    show_legend: bool,
    show_screen: bool,
) -> None:
    # Load trained model.
    model = load_trained_model(model_filepath)
    event_logger().info("Trained model is loaded: %s", model_filepath)

    # Load test datasets to calculate the score.
    event_logger().debug("Loading datasets with following condition:")
    event_logger().debug("     Path list: %s", dataset_paths)
    event_logger().debug("  glob pattern: %s", glob_pattern)
    event_logger().debug("        pickup: %s", n_pickup)
    datasets = load_datasets(dataset_paths, glob_pattern, n_pickup)
    event_logger().info("%s dataset files in total have been loaded", len(datasets))

    # Calculate prediction and score of the trained model based on test datasets.
    x_test, y_true = load_train_datasets(datasets, model.adapter)
    y_pred = model.predict(x_test)
    score = model.score(x_test, y_true)
    event_logger().info("Calculated score: %s", score)

    # Save calculated score.
    save_score(output_dir_path, output_prefix, score)

    # Make plots and save them.
    plot(output_dir_path, plot_prefix, model.adapter, y_true, y_pred, dpi=dpi, show_legend=show_legend, ext=plot_ext)

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
        "-n",
        "--n-pickup",
        type=int,
        help="Number of files to be loaded in each specified directory.",
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
        default="score",
        help="Filename prefix that will be added to calculated score.",
    )
    parser.add_argument(
        "--plot-prefix",
        default="prediction",
        help="Filename prefix that will be added to plot figure.",
    )
    parser.add_argument(
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
        default=True,
        help="Boolean. If True, show the plot figure. (default: True)",
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
        args.n_pickup,
        # parameters
        # output
        output_dir,
        args.output_prefix,
        args.plot_prefix,
        args.plot_ext,
        # boolean arguments
        dpi=dpi,
        show_legend=args.show_legend,
        show_screen=args.show_screen,
    )


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "cb csv dataset datasets env pred regressor usr vv"
# End:
