#!/usr/bin/env python

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING

from affetto_nn_ctrl.control_utility import (
    RobotInitializer,
    create_controller,
    release_pressure,
)
from affetto_nn_ctrl.data_handling import (
    build_data_dir_path,
    copy_config,
    get_default_base_dir,
    prepare_data_dir_path,
)
from affetto_nn_ctrl.event_logging import get_event_logger, start_event_logging

if TYPE_CHECKING:
    import logging

DEFAULT_DURATION = 10
APP_NAME_COLLECT_DATA = "initializer_test"


def run(
    config: str,
    sfreq: float | None,
    cfreq: float | None,
    init_duration: float | None,
    init_duration_keep_steady: float | None,
    init_manner: str | None,
    q_init: list[float] | None,
    ca_init: list[float] | None,
    cb_init: list[float] | None,
    duration: float,
) -> None:
    event_logger = get_event_logger()

    # Create controller and data logger.
    comm, ctrl, state = create_controller(config, sfreq, cfreq)
    if event_logger:
        event_logger.debug("Loading config file: %s", config)
        event_logger.debug("Controller created: sfreq=%s cfreq=%s", state.freq, ctrl.freq)

    # Initialize robot pose.
    initializer = RobotInitializer(
        ctrl.dof,
        config=config,
        duration=init_duration,
        duration_keep_steady=init_duration_keep_steady,
        manner=init_manner,
        q_init=q_init,
        ca_init=ca_init,
        cb_init=cb_init,
    )
    initializer.get_back_home((comm, ctrl, state))
    if event_logger:
        event_logger.debug("Initialized robot pose:")
        event_logger.debug("  config=%s", config)
        event_logger.debug(
            "  total_duration=%s (%s + %s), manner=%s",
            initializer.total_duration,
            initializer.duration,
            initializer.duration_keep_steady,
            initializer.get_manner(),
        )
        event_logger.info("Expected initialized pose: %s", initializer.get_q_init())
        event_logger.info("Final valve commands (ca): %s", initializer.get_ca_init())
        event_logger.info("Final valve commands (cb): %s", initializer.get_cb_init())
        event_logger.info("Actually initialized pose: %s", state.q)

    # Keep the initialized pose.
    if event_logger:
        event_logger.info("Keep the initialized pose for %s seconds...", duration)
    time.sleep(duration)

    # Release pressure.
    release_pressure((comm, ctrl, state))

    # Finish stuff.
    if event_logger:
        event_logger.debug("Test initializer finished")
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
        description="Test pose initializer.",
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
        "--init-duration-keep-steady",
        type=float,
        help="Time duration for keep steady after making the robot get back to home position.",
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
        help="Time duration for keeping the initialized pose.",
    )
    # Output
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory where collected dataset is stored.",
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


def start_logging(argv: list[str], output_dir: Path, verbose_count: int) -> logging.Logger:
    match verbose_count:
        case 0:
            logging_level = "WARNING"
        case 1:
            logging_level = "INFO"
        case _:
            logging_level = "DEBUG"

    return start_event_logging(argv, output_dir, name=__name__, logging_level=logging_level)


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
    prepare_data_dir_path(output_dir, make_latest_symlink=args.make_latest_symlink)
    copy_config(args.config, output_dir)
    event_logger = start_logging(sys.argv, output_dir, args.verbose)
    if event_logger:
        event_logger.info("Output directory: %s", output_dir)

    # Start mainloop
    run(
        # configuration
        args.config,
        args.sfreq,
        args.cfreq,
        args.init_duration,
        args.init_duration_keep_steady,
        args.init_manner,
        args.q_init,
        args.ca_init,
        args.cb_init,
        # input
        # parameters
        args.duration,
        # output
        # boolean arguments
    )


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "cb cfreq dataset dir env init sfreq sublabel symlink usr vv"
# End:
