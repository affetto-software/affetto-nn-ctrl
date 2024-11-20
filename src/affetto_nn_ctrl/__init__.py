from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

import numpy as np
from affctrllib import AffComm, AffPosCtrl, AffStateThread

ROOT_DIR_PATH = Path(__file__).parent.parent.parent
SRC_DIR_PATH = ROOT_DIR_PATH / "src"
APPS_DIR_PATH = ROOT_DIR_PATH / "apps"
TESTS_DIR_PATH = ROOT_DIR_PATH / "tests"
DEFAULT_BASE_DIR_PATH = ROOT_DIR_PATH / "data"
DEFAULT_CONFIG_PATH = ROOT_DIR_PATH / "config" / "affetto.toml"

DEFAULT_DURATION = 10.0  # sec
DEFAULT_SEED = None
DEFAULT_N_REPEAT = 1
DEFAULT_TIME_HOME = 10

CONTROLLER_T: TypeAlias = tuple[AffComm, AffPosCtrl, AffStateThread]
RefFuncType: TypeAlias = Callable[[float], np.ndarray]

# Local Variables:
# jinx-local-words: "src"
# End:
