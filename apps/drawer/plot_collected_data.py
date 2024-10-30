#!/usr/bin/env python

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from affetto_nn_ctrl import CONTROLLER_T, DEFAULT_SEED
from affetto_nn_ctrl.control_utility import (
    RandomTrajectory,
    RobotInitializer,
    control_position,
    create_controller,
    create_default_logger,
    release_pressure,
    resolve_joints_str,
)
from affetto_nn_ctrl.data_handling import (
    build_data_file_path,
    copy_config,
    get_default_base_dir,
    get_default_counter,
    get_output_dir_path,
    prepare_data_dir_path,
)
from affetto_nn_ctrl.event_logging import get_event_logger, start_logging

if TYPE_CHECKING:
    from pathlib import Path

    from affctrllib import Logger
