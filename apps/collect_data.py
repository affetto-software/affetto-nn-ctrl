#!/usr/bin/env python

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from affetto_nn_ctrl import DEFAULT_CONFIG_PATH, DEFAULT_SEED
from affetto_nn_ctrl.data_handling import get_default_base_dir

DEFAULT_DURATION = 10
DEFAULT_UPDATE_T_RANGE = (0.5, 1.5)
DEFAULT_UPDATE_Q_RANGE = (20.0, 40.0)
DEFAULT_Q_LIMIT = (5.0, 95.0)
DEFAULT_N_REPEAT = 1


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect data by generating random trajectories of specified joints.")
    # Configuration
    parser.add_argument(
        "-b",
        "--base-dir",
        default=get_default_base_dir(),
        help="Base directory of data location.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Config file path for robot model.",
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
        "-j",
        "--joint",
        nargs="*",
        type=int,
        help="Joint index list to make motion.",
    )
    # Input
    # Parameters
    parser.add_argument(
        "-T",
        "--time",
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
        dest="n",
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
        required=True,
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
    return parser.parse_args()


def main():
    args = parse()
    # mainloop(
    #     args.config,
    #     args.output,
    #     args.joint,
    #     args.time,
    #     args.seed,
    #     args.max_diff,
    #     args.time_range,
    #     args.limit,
    #     args.n,
    #     args.start_index,
    #     args.sfreq,
    #     args.cfreq,
    # )


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "cfreq dir env sfreq sublabel symlink usr"
# End:
