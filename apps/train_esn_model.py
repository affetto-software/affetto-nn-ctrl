#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from affetto_nn_ctrl import DEFAULT_SEED
from affetto_nn_ctrl.control_utility import (
    resolve_joints_str,
)
from affetto_nn_ctrl.data_handling import (
    build_data_file_path,
    copy_config,
    get_default_base_dir,
    get_output_dir_path,
    prepare_data_dir_path,
    train_test_split_files,
)
from affetto_nn_ctrl.event_logging import event_logger, start_logging
from affetto_nn_ctrl.model_utility import (
    dump_trained_model,
    load_data_adapter,
    load_datasets,
    load_esn_model,
    load_model_config_file,
    train_esn_model,
)

if TYPE_CHECKING:
    from pathlib import Path


APP_NAME_TRAIN_MODEL = "trained_model"
DEFAULT_DOF = 13


def run(
    joints_str: list[str] | None,
    dataset_paths: list[str],  # required
    glob_pattern: str,  # default: **/*.csv
    train_size: float | None,
    test_size: float | None,
    seed: int | None,
    model_config: str,  # required
    adapter_selector: str | None,
    scaler_selector: str | None,
    regressor_selector: str | None,
    output_dir_path: Path,
    output_prefix: str,
    *,
    shuffle: bool,
    split_in_each_directory: bool,
    overwrite: bool,
) -> None:
    # Resolve active joints.
    active_joints = resolve_joints_str(joints_str, dof=DEFAULT_DOF)
    event_logger().debug("Resolved active joints: %s", active_joints)

    # Load a model configuration file.
    config_dict = load_model_config_file(model_config)
    event_logger().debug("Model config file loaded: %s", model_config)

    # Create a data adapter.
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
    train_datasets = load_datasets(train_dataset_files)
    adapter = load_data_adapter(config_dict["model"]["adapter"], active_joints, adapter_selector)

    # Create an ESN model and train it.
    scaler, model = load_esn_model(config_dict["model"], active_joints, scaler_selector, regressor_selector)

    event_logger().info("Training model...")
    trained_model = train_esn_model(model, scaler, train_datasets, adapter)
    event_logger().debug("Training has done")

    # Save the trained model.
    suffix = ""
    if not overwrite:
        # Prevent a dumped file from being overwritten.
        n = len(list(output_dir_path.glob(f"{output_prefix}*.joblib")))
        if n > 0:
            suffix = f"_{n - 1:03d}"
    trained_model_file_path = build_data_file_path(
        output_dir_path,
        prefix=output_prefix + suffix,
        ext=".joblib",
    )
    dump_trained_model(trained_model, trained_model_file_path)
    event_logger().info("Trained model saved: %s", trained_model_file_path)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train regressor model with specified data adapter.",
    )
    default_base_dir = get_default_base_dir()
    # Configuration
    parser.add_argument(
        "-b",
        "--base-dir",
        default=str(default_base_dir),
        help="Base directory path for the current working project.",
    )
    parser.add_argument(
        "-j",
        "--joints",
        nargs="*",
        help="Active joint index list allowing to move.",
    )
    # Input
    parser.add_argument(
        "datasets",
        nargs="+",
        help="Path to files or directories which contain data sets used for training the model.",
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
    parser.add_argument(
        "-m",
        "--model-config",
        required=True,
        help="Config file path for regressor model and data adapter.",
    )
    parser.add_argument(
        "-a",
        "--adapter",
        help="Data adapter selector. Choose name and parameter set among those defined in model configuration file.",
    )
    parser.add_argument(
        "-s",
        "--scaler",
        help="Scaler selector. Choose name and parameter set among those defined in model configuration file.",
    )
    parser.add_argument(
        "-r",
        "--regressor",
        help="Regressor selector. Choose name and parameter set among those defined in model configuration file.",
    )
    # Output
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory where the trained model is stored.",
    )
    parser.add_argument(
        "--output-prefix",
        default="trained_model",
        help="Filename prefix that will be added to the trained model.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Boolean. If True, restart counter of generated data files and overwrite existing files.",
    )
    parser.add_argument(
        "--label",
        default="testing",
        help="Label name of the current dataset.",
    )
    parser.add_argument(
        "--sublabel",
        help="Optional. Sublabel string for the current dataset.",
    )
    parser.add_argument(
        "--split-by-date",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Boolean. If True, split generated dataset by date.",
    )
    parser.add_argument(
        "--specify-date",
        help="Specify date string like '20240123T123456' or 'latest'. When the date string is specified, "
        "generated dataset will be stored in the specified date directory. When 'latest' is specified, "
        "find the latest directory.",
    )
    parser.add_argument(
        "--make-latest-symlink",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Boolean. If True, make a symbolic link to the latest.",
    )
    # Others
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Enable verbose console output. -v provides additional info. -vv provides debug output.",
    )
    return parser.parse_args()


def main() -> None:

    args = parse()
    if args.train_size is not None and args.train_size > 1:
        args.train_size = int(args.train_size)
    if args.test_size is not None and args.test_size > 1:
        args.test_size = int(args.test_size)

    # Prepare input/output
    output_dir = get_output_dir_path(
        args.base_dir,
        APP_NAME_TRAIN_MODEL,
        args.output,
        args.label,
        args.sublabel,
        args.specify_date,
        split_by_date=args.split_by_date,
    )
    start_logging(sys.argv, output_dir, __name__, args.verbose)
    event_logger().info("Output directory: %s", output_dir)
    prepare_data_dir_path(output_dir, make_latest_symlink=args.make_latest_symlink)
    copy_config(None, None, args.model_config, output_dir)
    event_logger().debug("Parsed arguments: %s", args)

    # Start mainloop
    run(
        # configuration
        args.joints,
        # input
        args.datasets,
        args.glob_pattern,
        args.train_size,
        args.test_size,
        args.seed,
        # parameters
        args.model_config,
        args.adapter,
        args.scaler,
        args.regressor,
        # output
        output_dir,
        args.output_prefix,
        # boolean arguments
        shuffle=args.shuffle,
        split_in_each_directory=args.split_in_each_directory,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "csv dataset datasets dir env joblib regressor scaler sublabel symlink usr vv"
# End:
