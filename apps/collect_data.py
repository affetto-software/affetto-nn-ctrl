#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

from affetto_nn_ctrl import DEFAULT_SEED
from affetto_nn_ctrl.control_utility import GetGlobalRngT
from affetto_nn_ctrl.data_handling import build_data_dir_path, get_default_base_dir, prepare_data_dir_path
from affetto_nn_ctrl.event_logging import get_event_logger, start_event_logging

DEFAULT_DURATION = 10
DEFAULT_UPDATE_T_RANGE = (0.5, 1.5)
DEFAULT_UPDATE_Q_RANGE = (20.0, 40.0)
DEFAULT_Q_LIMIT = (5.0, 95.0)
DEFAULT_N_REPEAT = 1
APP_NAME_COLLECT_DATA = "dataset"


def run(
    config: str,
    joint: list[int],
    sfreq: float | None,
    cfreq: float | None,
    duration: float,
    t_range: tuple[float, float],
    q_range: tuple[float, float],
    q_limit: tuple[float, float],
    n_repeat: int,
    output_dir_path: Path,
    seed: int | GetGlobalRngT | None,
    *,
    overwrite: bool,
) -> None:
    event_logger = get_event_logger()
    if event_logger:
        msg = f"Config: {config}"
        event_logger.debug(msg)


def make_output_dir(
    base_dir: str,
    output: str | None,
    label: str | None,
    sublabel: str | None,
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
            split_by_date=split_by_date,
            millisecond=False,
        )
    return output_dir_path


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect data by generating random trajectories of specified joints.")
    default_base_dir = get_default_base_dir()
    # Configuration
    parser.add_argument(
        "-b",
        "--base-dir",
        default=str(default_base_dir),
        help="Base directory of data location.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(default_base_dir / "config/affetto.toml"),
        help="Config file path for robot model.",
    )
    parser.add_argument(
        "-j",
        "--joint",
        nargs="*",
        type=int,
        help="Joint index list to make motion.",
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
    # Input
    # Parameters
    parser.add_argument(
        "-T",
        "--duration",
        default=DEFAULT_DURATION,
        type=float,
        help="Time duration for each trajectory.",
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
        help="Time range when updating each joint angle reference.",
    )
    parser.add_argument(
        "-q",
        "--q-range",
        default=DEFAULT_UPDATE_Q_RANGE,
        nargs="+",
        type=float,
        help="Joint angle range when updating.",
    )
    parser.add_argument(
        "-Q",
        "--q-limit",
        default=DEFAULT_Q_LIMIT,
        nargs="+",
        type=float,
        help="Joint angle limit.",
    )
    parser.add_argument(
        "-n",
        "--n-repeat",
        default=DEFAULT_N_REPEAT,
        type=int,
        help="Number of iterations to generate reference trajectories.",
    )
    # Output
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory where collected data files are stored.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Boolean. If True, overwrite existing reference trajectories.",
    )
    parser.add_argument(
        "--label",
        help="Label string for generated data.",
    )
    parser.add_argument(
        "--sublabel",
        help="Optional. Sublabel string for generated data.",
    )
    parser.add_argument(
        "--split-by-date",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Boolean. If True, split outputs by date.",
    )
    parser.add_argument(
        "--make-latest-symlink",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Boolean. If True, make a symbolic link to the latest.",
    )
    # Others
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Make console output verbose.")
    return parser.parse_args()


def main() -> None:
    import sys

    args = parse()

    # Prepare input/output
    match args.verbose:
        case 0:
            logging_level = "WARNING"
        case 1:
            logging_level = "INFO"
        case _:
            logging_level = "DEBUG"

    output_dir = make_output_dir(
        args.base_dir,
        args.output,
        args.label,
        args.sublabel,
        split_by_date=args.split_by_date,
    )
    start_event_logging(sys.argv, output_dir, name=__name__, logging_level=logging_level)
    event_logger = get_event_logger()
    if event_logger:
        msg = f"Output directory: {output_dir}"
        event_logger.info(msg)
    prepare_data_dir_path(output_dir, make_latest_symlink=args.make_latest_symlink, dry_run=True)

    # Start mainloop
    run(
        # configuration
        args.config,
        args.joint,
        args.sfreq,
        args.cfreq,
        # input
        # parameters
        args.duration,
        args.seed,
        args.t_range,
        args.q_range,
        args.q_limit,
        args.n_repeat,
        # output
        output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "cfreq dataset dir env sfreq sublabel symlink usr"
# End:
