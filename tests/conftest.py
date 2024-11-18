from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Any

import pytest

from affetto_nn_ctrl import TESTS_DIR_PATH

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture(scope="module")
def make_work_directory() -> Generator[Path, Any, Any]:
    work_dir = TESTS_DIR_PATH / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    yield work_dir
    shutil.rmtree(work_dir)
