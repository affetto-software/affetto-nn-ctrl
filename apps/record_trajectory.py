#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from affctrllib import Logger

from affetto_nn_ctrl import CONTROLLER_T
from affetto_nn_ctrl.control_utility import (
    RobotInitializer,
    create_controller,
    create_default_logger,
    reset_logger,
    resolve_joints_str,
)
from affetto_nn_ctrl.data_handling import (
    build_data_dir_path,
    get_default_base_dir,
    get_default_counter,
    prepare_data_dir_path,
)
from affetto_nn_ctrl.event_logging import get_event_logger, start_event_logging

DEFAULT_DURATION = 10
APP_NAME_COLLECT_DATA = "reference_trajectory"


def record_motion(
    controller: CONTROLLER_T,
    data_logger: Logger,
    active_joints: list[int],
    duration: float,
    data_file_path: Path,
    header_text: str,
) -> None:
    reset_logger(data_logger, data_file_path)
    comm, ctrl, state = controller
    ca, cb = np.zeros(ctrl.dof, dtype=float), np.zeros(ctrl.dof, dtype=float)


def run(
    config: str,
    joints_str: list[str] | None,
    sfreq: float | None,
    cfreq: float | None,
    init_duration: float | None,
    init_manner: str | None,
    q_init: list[float] | None,
    ca_init: list[float] | None,
    cb_init: list[float] | None,
    duration: float,
    output_dir_path: Path,
    output_prefix: str,
    *,
    overwrite: bool,
) -> None:
    event_logger = get_event_logger()

    # Create controller and data logger.
    comm, ctrl, state = create_controller(config, sfreq, cfreq)
    data_logger = create_default_logger(ctrl.dof)
    if event_logger:
        event_logger.debug("Loading config file: %s", config)
        event_logger.debug("Controller created: sfreq=%s cfreq=%s", state.freq, ctrl.freq)
        event_logger.debug("Default logger created: DOF=%s", ctrl.dof)

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
    q0 = state.q
    if initializer.get_manner() == "position":
        q0 = initializer.get_q_init()
    t0 = 0.0
    if event_logger:
        event_logger.debug("Initializer created: manner=%s", initializer.get_manner())
        event_logger.debug("Initial posture: %s", q0)

    # Resolve active joints.
    active_joints = resolve_joints_str(joints_str)
    if event_logger:
        event_logger.debug("Resolved active joints: %s", active_joints)

    # Create data file counter.
    n = 0
    if not overwrite:
        n = len(list(output_dir_path.glob(f"{output_prefix}*.csv")))
    cnt = get_default_counter(n)
    if event_logger:
        event_logger.debug("Data file counter initialized with %s", n)


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
        description="Record motion reference trajectory by kinesthetic teaching.",
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
        help="Active joint index list allowing to move.",
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
    # Output
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory where collected dataset is stored.",
    )
    parser.add_argument(
        "--output-prefix",
        default="reference_trajectory",
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
        # output
        output_dir,
        args.output_prefix,
        # boolean arguments
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "cb cfreq csv dataset dir env init sfreq sublabel symlink usr vv"
# End:
