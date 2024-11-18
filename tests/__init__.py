from __future__ import annotations

from typing import TYPE_CHECKING

from affetto_nn_ctrl import TESTS_DIR_PATH

if TYPE_CHECKING:
    from pathlib import Path

TESTS_DATA_DIR_PATH = TESTS_DIR_PATH / "data"


def assert_file_contents(expected: Path, actual: Path) -> None:
    from difflib import unified_diff

    with expected.open() as f:
        expected_lines = f.readlines()
    with actual.open() as f:
        actual_lines = f.readlines()
    diff = list(unified_diff(expected_lines, actual_lines))
    assert diff == [], "Unexpected file differences:\n" + "".join(diff)
