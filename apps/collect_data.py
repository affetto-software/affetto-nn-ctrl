#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from affetto_nn_ctrl import CONTROLLER_T, DEFAULT_SEED
from affetto_nn_ctrl.control_utility import (
    RandomTrajectory,
    RobotInitializer,
    control_position,
    create_controller,
    create_default_logger,
)
from affetto_nn_ctrl.data_handling import (
    build_data_dir_path,
    build_data_file_path,
    get_default_base_dir,
    get_default_counter,
    prepare_data_dir_path,
)
from affetto_nn_ctrl.event_logging import get_event_logger, start_event_logging

if TYPE_CHECKING:
    from affctrllib import Logger

DEFAULT_DURATION = 10
DEFAULT_UPDATE_T_RANGE = (0.5, 1.5)
DEFAULT_UPDATE_Q_RANGE = (20.0, 40.0)
DEFAULT_UPDATE_Q_LIMIT = (5.0, 95.0)
DEFAULT_UPDATE_PROFILE = "trapezoidal"
DEFAULT_N_REPEAT = 5
APP_NAME_COLLECT_DATA = "dataset"


def record_data(
    controller: CONTROLLER_T,
    data_logger: Logger,
    rt: RandomTrajectory,
    duration: float,
    data_file_path: Path,
    header_text: str,
) -> None:
    qdes_func, dqdes_func = rt.get_qdes_func(), rt.get_dqdes_func()
    control_position(
        controller,
        qdes_func,
        dqdes_func,
        duration,
        data_logger,
        data_file_path,
        time_updater="accumulated",
        header_text=header_text,
    )


def run(
    config: str,
    joints: list[int],
    sfreq: float | None,
    cfreq: float | None,
    init_duration: float | None,
    init_manner: str | None,
    q_init: list[float] | None,
    ca_init: list[float] | None,
    cb_init: list[float] | None,
    duration: float,
    t_range: tuple[float, float],
    q_range: tuple[float, float],
    q_limit: tuple[float, float],
    profile: str,
    n_repeat: int,
    seed: int | None,
    output_dir_path: Path,
    output_prefix: str,
    *,
    async_mode: bool,
    overwrite: bool,
) -> None:
    event_logger = get_event_logger()
    if event_logger:
        event_logger.debug("Loading config file: %s", config)

    # Create controller and data logger.
    comm, ctrl, state = create_controller(config, sfreq, cfreq)
    data_logger = create_default_logger(ctrl.dof)

    # Initialize robot pose.
    initializer = RobotInitializer(
        ctrl.dof,
        config=config,
        duration=init_duration,
        manner=init_manner,
        q_init=q_init,
        ca_init=ca_init,
        cb_init=cb_init,
    )
    initializer.get_back_home((comm, ctrl, state))

    # Create random trajectory generator.
    q0 = state.q
    if initializer.get_manner() == "position":
        q0 = initializer.get_q_init()
    t0 = 0.0
    rt = RandomTrajectory(joints, q0, t0, t_range, q_range, q_limit, profile, seed, async_update=async_mode)

    # Generate dataset.
    if not overwrite:
        n = len(list(output_dir_path.glob(f"{output_prefix}*.csv")))
        cnt = get_default_counter(n)
    else:
        cnt = get_default_counter()
    for i in range(n_repeat):
        data_file_path = build_data_file_path(output_dir_path, prefix=output_prefix, iterator=cnt, ext=".csv")
        header_text = f"[{i+1}/{n_repeat}] Collecting motion data..."
        record_data(
            (comm, ctrl, state),
            data_logger,
            rt,
            duration,
            data_file_path,
            header_text=header_text,
        )
        # Prepare for next record.
        initializer.get_back_home((comm, ctrl, state))

    # Finish stuff.
    comm.close_command_socket()
    state.join()


def make_output_dir(
    base_dir: str,
    output: str | None,
    label: str | None,
    sublabel: str | None,
    specified_date: str | None,
    *,
    split_by_date: bool,
) -> Path:
    output_dir_path: Path
    if output is not None:
        output_dir_path = Path(output)
    else:
        if label is None:
            label = "testing"
        output_dir_path = build_data_dir_path(
            base_dir,
            APP_NAME_COLLECT_DATA,
            label,
            sublabel,
            specified_date,
            split_by_date=split_by_date,
            millisecond=False,
        )
    return output_dir_path


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect data by letting specified joints track randomly generated trajectories.",
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
        "-c",
        "--config",
        default=str(default_base_dir / "config/affetto.toml"),
        help="Config file path for robot model.",
    )
    parser.add_argument(
        "-j",
        "--joints",
        nargs="*",
        type=int,
        help="Active joint index list to make motion.",
    )
    parser.add_argument(
        "-F",
        "--sensor-freq",
        dest="sfreq",
        type=float,
        help="Sensor frequency.",
    )
    parser.add_argument(
        "-f",
        "--control-freq",
        dest="cfreq",
        type=float,
        help="Control frequency.",
    )
    parser.add_argument(
        "--init-duration",
        type=float,
        help="Time duration for making the robot get back to home position.",
    )
    parser.add_argument(
        "--init-manner",
        help="How to make the robot get back to the home position. Choose: [position, pressure]",
    )
    parser.add_argument(
        "--q-init",
        nargs="+",
        type=float,
        help="Joint angle list when making the robot get back to home position.",
    )
    parser.add_argument(
        "--ca-init",
        nargs="+",
        type=float,
        help="Valve command list for positive side when making the robot get back to home position.",
    )
    parser.add_argument(
        "--cb-init",
        nargs="+",
        type=float,
        help="Valve command list for negative side when making the robot get back to home position.",
    )
    # Input
    # Parameters
    parser.add_argument(
        "-T",
        "--duration",
        default=DEFAULT_DURATION,
        type=float,
        help="Time duration of generated trajectory.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=DEFAULT_SEED,
        help="Seed value given to random number generator.",
    )
    parser.add_argument(
        "-t",
        "--t-range",
        default=DEFAULT_UPDATE_T_RANGE,
        nargs="+",
        type=float,
        help="Time range when updating joint angle references.",
    )
    parser.add_argument(
        "-q",
        "--q-range",
        default=DEFAULT_UPDATE_Q_RANGE,
        nargs="+",
        type=float,
        help="Joint angle range when updating joint angle references.",
    )
    parser.add_argument(
        "-Q",
        "--q-limit",
        default=DEFAULT_UPDATE_Q_LIMIT,
        nargs="+",
        type=float,
        help="Joint angle limit when generating joint angle references.",
    )
    parser.add_argument(
        "-p",
        "--profile",
        default=DEFAULT_UPDATE_PROFILE,
        help="Update profile for joint angle references. Typically, 'trapezoidal' or 'step' is used.",
    )
    parser.add_argument(
        "--async-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Boolean. If True, asynchronous update model is enabled. "
        "Otherwise, references are updated synchronously.",
    )
    parser.add_argument(
        "-n",
        "--n-repeat",
        default=DEFAULT_N_REPEAT,
        type=int,
        help="Number of iterations to generate trajectories.",
    )
    # Output
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory where collected dataset is stored.",
    )
    parser.add_argument(
        "--output-prefix",
        default="motion_data_",
        help="Filename prefix that will be added to generated data file.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Boolean. If True, restart counter of generated data files and overwrite existing data files.",
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


def start_logging(argv: list[str], output_dir: Path, verbose_count: int) -> None:
    match verbose_count:
        case 0:
            logging_level = "WARNING"
        case 1:
            logging_level = "INFO"
        case _:
            logging_level = "DEBUG"

    start_event_logging(argv, output_dir, name=__name__, logging_level=logging_level)


def main() -> None:
    import sys

    args = parse()

    # Prepare input/output
    output_dir = make_output_dir(
        args.base_dir,
        args.output,
        args.label,
        args.sublabel,
        args.specify_date,
        split_by_date=args.split_by_date,
    )
    start_logging(sys.argv, output_dir, args.verbose)
    event_logger = get_event_logger()
    if event_logger:
        event_logger.info("Output directory: %s", output_dir)
    prepare_data_dir_path(output_dir, make_latest_symlink=args.make_latest_symlink, dry_run=True)

    # Start mainloop
    run(
        # configuration
        args.config,
        args.joints,
        args.sfreq,
        args.cfreq,
        args.init_duration,
        args.init_manner,
        args.q_init,
        args.ca_init,
        args.cb_init,
        # input
        # parameters
        args.duration,
        args.t_range,
        args.q_range,
        args.q_limit,
        args.profile,
        args.n_repeat,
        args.seed,
        # output
        output_dir,
        args.output_prefix,
        # boolean arguments
        async_mode=args.async_mode,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "cb cfreq csv dataset dir env init noqa sfreq sublabel symlink usr vv"
# End:
