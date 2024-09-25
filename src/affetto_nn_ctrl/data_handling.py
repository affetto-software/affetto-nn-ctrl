from __future__ import annotations

import datetime
import itertools
from pathlib import Path
from typing import TYPE_CHECKING

from affetto_nn_ctrl import APPS_DIR_PATH, DEFAULT_BASE_DIR_PATH

if TYPE_CHECKING:
    from collections.abc import Iterator


def get_default_base_dir() -> Path:
    base_dir_config = APPS_DIR_PATH / "base_dir"
    if base_dir_config.exists():
        return Path(base_dir_config.read_text())
    return DEFAULT_BASE_DIR_PATH


def get_default_counter(start: int = 0, step: int = 1, fmt: str = "{:03d}") -> Iterator:
    return map(fmt.format, itertools.count(start, step))


def build_data_dir_path(
    base_dir: str | Path | None = None,
    app_name: str = "app",
    label: str = "test",
    sublabel: str | None = None,
    *,
    split_by_date: bool = True,
    millisecond: bool = False,
) -> Path:
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR_PATH

    # Build upon the provided app name and label.
    built_path: Path = Path(base_dir) / app_name / label

    # Split directory into sub-directory by date.
    if split_by_date:
        fmt = "%Y%m%dT%H%M%S"
        if millisecond:
            fmt += ".%f"
        now = datetime.datetime.now().strftime(fmt)  # noqa: DTZ005
        built_path /= now

    # Add the provided sublabel.
    if sublabel is not None and len(sublabel) > 0:
        built_path /= sublabel

    return built_path


def build_data_file_path(
    output_dir: str | Path,
    prefix: str = "",
    iterator: Iterator | None = None,
    ext: str | None = None,
) -> Path:
    built_path: Path = Path(output_dir)

    # When ext is provided, generate a specific filename with prefix and iterator.
    if ext is not None:
        if len(ext) > 0:
            ext = f".{ext}" if not ext.startswith(".") else ext
        suffix = next(iterator) if iterator else ""
        filename = f"{prefix}{suffix}{ext}"
        if len(filename) == len(ext):
            msg = "Extension was given, but unable to determine filename"
            raise ValueError(msg)
        built_path /= filename

    return built_path


# Local Variables:
# jinx-local-words: "dT dir noqa sublabel"
# End:
