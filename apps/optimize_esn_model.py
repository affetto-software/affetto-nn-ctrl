#!/usr/bin/env python
# ruff: noqa: N806
from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

import numpy as np
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error

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
from affetto_nn_ctrl.esn import ESN
from affetto_nn_ctrl.event_logging import event_logger, start_logging
from affetto_nn_ctrl.model_utility import (
    DataAdapterBase,
    DelayStates,
    DelayStatesAll,
    PreviewRef,
    dump_trained_model,
    load_data_adapter,
    load_datasets,
    load_esn_model,
    load_model_config_file,
    load_scaler,
    load_train_datasets,
    train_esn_model,
)
from affetto_nn_ctrl.random_utility import set_seed

if TYPE_CHECKING:
    from pathlib import Path

    from optuna.study import Study

    from affetto_nn_ctrl._typing import Unknown


APP_NAME_OPTIMIZE = "optimization"
DEFAULT_DOF = 13


def optimize(
    n_trials: int,
    config: dict[str, Unknown],
    train_dataset_files: list[Path],
    test_dataset_files: list[Path],
    adapter: DataAdapterBase,
    scaler_selectors: list[str],
    active_joints: list[int],
    seed: int | None,
) -> Study:
    train_datasets = load_datasets(train_dataset_files)
    x_train, y_train = load_train_datasets(train_datasets, adapter)
    test_datasets = load_datasets(test_dataset_files)
    loaded_test_datasets = [load_train_datasets(test_dataset, adapter) for test_dataset in test_datasets]

    # Default parameters
    N_u = len(active_joints) * 5
    N_y = len(active_joints) * 2

    def objective(trial: Trial) -> float | Unknown:
        scaler_selector = trial.suggest_categorical("scaler_selector", scaler_selectors)
        N_x = trial.suggest_int("N_x", 400, 3000, step=100)
        density = trial.suggest_float("density", 0.01, 0.5)
        rho = trial.suggest_float("rho", 0.1, 2.0)
        leaking_rate = trial.suggest_float("leaking_rate", 0.01, 1.0)
        input_scale = trial.suggest_float("input_scale", 0.01, 1.0)

        scaler = load_scaler(config["scaler"], scaler_selector)
        esn = ESN(
            N_u=N_u,
            N_y=N_y,
            N_x=N_x,
            density=density,
            rho=rho,
            leaking_rate=leaking_rate,
            input_scale=input_scale,
        )
        if scaler is not None:
            x_train_fit = scaler.fit_transform(x_train)
        else:
            x_train_fit = x_train
        esn.fit(x_train_fit, y_train)

        mse: list[float | Unknown] = []
        for x_test, y_true in loaded_test_datasets:
            y_pred = esn.predict(x_test)
            mse.append(mean_squared_error(y_true, y_pred))
        return np.mean(mse)

    sampler = TPESampler(seed=seed) if seed is not None else None
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study


def _toml_string(value: Unknown) -> str:
    if value is None:
        value = "None"
    elif isinstance(value, bool):
        return "true" if value else "false"
    if value is np.tanh:
        value = "tanh"

    if isinstance(value, str):
        return f'"{value}"'
    return f"{value}"


def adapter_name(adapter: DataAdapterBase) -> str:
    if isinstance(adapter, PreviewRef):
        return "preview-ref"
    if isinstance(adapter, DelayStates):
        return "delay-states"
    if isinstance(adapter, DelayStatesAll):
        return "delay-states-all"
    msg = f"invalid adapter type, {type(adapter)}"
    raise ValueError(msg)


def save_optimization_result(
    output_dir_path: Path,
    output_prefix: str,
    study: Study,
    adapter: DataAdapterBase,
    seed: int | None,
    *,
    ext: str = ".toml",
    overwrite: bool = False,
) -> Path:
    suffix: str = ""
    if not overwrite:
        # Prevent a dumped file from being overwritten.
        n = len(list(output_dir_path.glob(f"{output_prefix}*{ext}")))
        if n > 0:
            suffix = f"_{n - 1:03d}"
    output = build_data_file_path(
        output_dir_path,
        prefix=output_prefix + suffix,
        ext=ext,
    )

    best_params = study.best_params.copy()
    best_scaler = best_params.pop("scaler_selector")
    text_lines = [
        "[optimization]\n",
        f"best_scaler = {_toml_string(best_scaler)}\n",
        f"best_value = {study.best_value}\n",
        f"trials = {len(study.trials)}\n",
        f"seed = {_toml_string(seed)}\n",
        "\n",
    ]

    if isinstance(adapter, PreviewRef | DelayStates | DelayStatesAll):
        text_lines.extend(
            [
                "[model]\n",
                "[model.adapter]\n",
                f'name = "{adapter_name(adapter)}"\n',
                'params = "default"\n',
                f"active_joints = [{', '.join(map(str, adapter.params.active_joints))}]\n",
                f"dt = {adapter.params.dt}\n",
                f"include_dqdes = {_toml_string(adapter.params.include_dqdes)}\n",
                f"ctrl_step = {_toml_string(adapter.params.ctrl_step)}\n",
                "\n",
            ],
        )
    if isinstance(adapter, PreviewRef):
        text_lines.extend(
            [
                f"[model.adapter.{adapter_name(adapter)}.step{adapter.params.preview_step:02}]\n",
                f"preview_step = {adapter.params.preview_step}\n",
                f"[model.adapter.{adapter_name(adapter)}.default]\n",
                f"preview_step = {adapter.params.preview_step}\n",
                "\n",
            ],
        )
    elif isinstance(adapter, DelayStates | DelayStatesAll):
        text_lines.extend(
            [
                f"[model.adapter.{adapter_name(adapter)}.step{adapter.params.delay_step:02}]\n",
                f"delay_step = {adapter.params.delay_step}\n",
                f"[model.adapter.{adapter_name(adapter)}.default]\n",
                f"delay_step = {adapter.params.delay_step}\n",
                "\n",
            ],
        )

    text_lines.extend(
        [
            "[model.scaler]\n",
            f"name = {_toml_string(best_scaler)}\n",
            f"params = {_toml_string('default')}\n",
            "\n",
            f"[model.scaler.{best_scaler}.default]\n",
            "\n",
        ],
    )

    text_lines.extend(
        [
            "[model.regressor]\n",
            f"name = {_toml_string('esn')}\n",
            f"params = {_toml_string('best')}\n",
            "\n",
            "[model.regressor.esn.best]\n",
        ],
    )
    text_lines.extend([f"{key} = {_toml_string(value)}\n" for key, value in best_params.items()])
    text_lines.append("\n")

    with output.open("w") as f:
        f.writelines(text_lines)
    return output


def run(
    joints_str: list[str] | None,
    dataset_paths: list[str],  # required
    n_trials: int,  # default: 100
    glob_pattern: str,  # default: **/*.csv
    train_size: float | None,
    test_size: float | None,
    seed: int | None,
    opt_seed: int | None,
    model_config: str,  # required
    adapter_selector: str | None,
    scaler_selectors: list[str],
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
    adapter = load_data_adapter(config_dict["model"]["adapter"], active_joints, adapter_selector)

    # Optimize parameters
    event_logger().info("Optimizing parameters...")
    study = optimize(
        n_trials,
        config_dict["model"],
        train_dataset_files,
        test_dataset_files,
        adapter,
        scaler_selectors,
        active_joints,
        opt_seed,
    )
    event_logger().debug("Optimization has done")

    # Save the best parameters.
    output = save_optimization_result(
        output_dir_path,
        output_prefix,
        study,
        adapter,
        opt_seed,
        ext=".toml",
        overwrite=overwrite,
    )
    event_logger().info("Optimization result saved: %s", output)

    # Calculate score.
    config_dict = load_model_config_file(output)
    event_logger().debug("Best model config file loaded: %s", output)
    adapter = load_data_adapter(config_dict["model"]["adapter"], active_joints, adapter_selector)

    scaler_selector = study.best_params["scaler_selector"]
    regressor_selector = "esn.best"
    scaler, model = load_esn_model(config_dict["model"], active_joints, scaler_selector, regressor_selector)
    event_logger().info("Training best model...")
    train_datasets = load_datasets(train_dataset_files)
    trained_model = train_esn_model(model, scaler, train_datasets, adapter)
    event_logger().debug("Training has done")

    # Save the trained best model.
    output_prefix = "trained_model"
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
    event_logger().info("Trained best model saved: %s", trained_model_file_path)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize model parameters with specified data adapter.",
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
        "-n",
        "--trials",
        default=100,
        type=int,
        help="Number of trials for optimization process.",
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
        "--opt-seed",
        default=DEFAULT_SEED,
        type=int,
        help="Seed value given to optimization sampler.",
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
        nargs="+",
        default=["none"],
        help="List of scaler selector candidates.",
    )
    # Output
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory where the optimization result is stored.",
    )
    parser.add_argument(
        "--output-prefix",
        default="optimization",
        help="Filename prefix that will be added to the optimization result.",
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
        APP_NAME_OPTIMIZE,
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
    if args.opt_seed is not None:
        set_seed(args.opt_seed)
    run(
        # configuration
        args.joints,
        # input
        args.datasets,
        args.trials,
        args.glob_pattern,
        args.train_size,
        args.test_size,
        args.seed,
        args.opt_seed,
        # parameters
        args.model_config,
        args.adapter,
        args.scaler,
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
# jinx-local-words: "csv ctrl dir dqdes dt env esn joblib minmax noqa params regressor scaler sublabel symlink tanh usr vv" # noqa: E501
# End:
