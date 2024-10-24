#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

from affetto_nn_ctrl import DEFAULT_SEED
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
    seed: int | None,
    output_dir_path: Path,
    *,
    overwrite: bool,
) -> None:
    event_logger = get_event_logger()
    if event_logger:
        event_logger.debug("Config: %s", config)


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
        "--joint",
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
        default=DEFAULT_Q_LIMIT,
        nargs="+",
        type=float,
        help="Joint angle limit when generating joint angle references.",
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
        args.joint,
        args.sfreq,
        args.cfreq,
        # input
        # parameters
        args.duration,
        args.t_range,
        args.q_range,
        args.q_limit,
        args.n_repeat,
        args.seed,
        # output
        output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "cfreq dataset dir env sfreq sublabel symlink usr vv"
# End:
