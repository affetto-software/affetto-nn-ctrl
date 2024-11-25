from __future__ import annotations

import datetime
import itertools
import os
import re
import shutil
import warnings
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

from affetto_nn_ctrl import APPS_DIR_PATH, DEFAULT_BASE_DIR_PATH
from affetto_nn_ctrl.event_logging import event_logger

if TYPE_CHECKING:
    from collections.abc import Iterator


def get_default_base_dir(base_dir_config: Path | None = None) -> Path:
    if base_dir_config is None:
        base_dir_config = APPS_DIR_PATH / "base_dir"
    if base_dir_config.exists():
        return Path(base_dir_config.read_text().strip())
    return DEFAULT_BASE_DIR_PATH


def get_default_counter(start: int = 0, step: int = 1, fmt: str = "_{:03d}") -> Iterator:
    return map(fmt.format, itertools.count(start, step))


def build_data_dir_path(
    base_dir: str | Path | None = None,
    app_name: str = "app",
    label: str = "test",
    sublabel: str | None = None,
    specified_date: str | None = None,
    *,
    split_by_date: bool = True,
    millisecond: bool = False,
) -> Path:
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR_PATH

    # Build upon the provided app name and label.
    built_path: Path = Path(base_dir) / app_name / label

    # Split directory into sub-directory by date.
    if specified_date is not None:
        if specified_date == "latest":
            built_path = find_latest_data_dir_path(base_dir, app_name, label)
        else:
            built_path /= specified_date
    elif split_by_date:
        fmt = "%Y%m%dT%H%M%S"
        if millisecond:
            fmt += ".%f"
        now = datetime.datetime.now().strftime(fmt)  # noqa: DTZ005
        built_path /= now

    # Add the provided sublabel.
    if sublabel is not None and len(sublabel) > 0:
        built_path /= sublabel

    return built_path


def get_output_dir_path(
    base_dir: str,
    app_name: str,
    given_output: str | None,
    label: str | None,
    sublabel: str | None,
    specified_date: str | None,
    *,
    split_by_date: bool,
    millisecond: bool = False,
) -> Path:
    output_dir_path: Path
    if given_output is not None:
        output_dir_path = Path(given_output)
    else:
        if label is None:
            label = "testing"
        output_dir_path = build_data_dir_path(
            base_dir,
            app_name,
            label,
            sublabel,
            specified_date,
            split_by_date=split_by_date,
            millisecond=millisecond,
        )
    return output_dir_path


def split_data_dir_path_by_date(data_dir_path: Path) -> tuple[Path, str | None, str | None]:
    path = data_dir_path
    parts: list[str] = []
    date_pattern = re.compile(r"^[0-9]{8}T[0-9]{6}(?:\.[0-9]{6})?$")
    while path.name != "":
        name = path.name
        path = path.parent
        if date_pattern.match(name) is not None:
            return path, name, "/".join(reversed(parts))
        parts.append(name)
    return data_dir_path, None, None


def _make_latest_symlink(path: Path) -> None:
    path_head, date, _ = split_data_dir_path_by_date(path)
    if date is None:
        msg = f"Trying to make latest symlink, but no date part has found: {path}"
        event_logger().warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
    else:
        symlink_src = path_head / date
        symlink_path = path_head / "latest"
        if not symlink_path.exists() or symlink_path.is_symlink():
            try:
                os.remove(symlink_path)  # noqa: PTH107
            except OSError:
                pass
            finally:
                dst = symlink_src.absolute()
                os.symlink(dst, symlink_path)
                event_logger().debug("Symlink created: %s -> %s", symlink_path, dst)


def prepare_data_dir_path(
    data_dir_path: str | Path,
    *,
    make_latest_symlink: bool = False,
) -> Path:
    path = Path(data_dir_path)
    path.mkdir(parents=True, exist_ok=True)
    event_logger().debug("Directory created: %s", path)

    if make_latest_symlink:
        _make_latest_symlink(path)
    return path


def copy_config(
    config: str | Path | None,
    init_config: str | Path | None,
    model_config: str | Path | None,
    output_dir: Path,
) -> None:
    target_dir_path = output_dir / "config"
    prepare_data_dir_path(target_dir_path, make_latest_symlink=False)
    for file, kind in zip((config, init_config, model_config), ("Robot", "Initializer", "Model"), strict=True):
        if file is not None:
            shutil.copy(file, target_dir_path)
            event_logger().debug("%s config copied: %s -> %s", kind, config, target_dir_path)


def is_latest_data_dir_path_maybe(dirpath: Path, pattern: str = "20*T*") -> bool:
    if dirpath.name == "latest":
        return True
    return fnmatch(dirpath.name, pattern)


def find_latest_data_dir_path(
    base_dir: str | Path,
    app_name: str | None = None,
    label: str | None = None,
    glob_pattern: str = "**/20*T*",
    *,
    force_find_by_pattern: bool = False,
) -> Path:
    if app_name is not None and label is not None:
        search_dir_path = Path(base_dir) / app_name / label
    elif app_name is None and label is None:
        search_dir_path = Path(base_dir)
    else:
        msg = "Invalid base directory specification"
        raise RuntimeError(msg)

    symlink = search_dir_path / "latest"
    if not force_find_by_pattern and symlink.is_symlink() and symlink.exists():
        return symlink.resolve()
    sorted_dirs = sorted(search_dir_path.glob(glob_pattern), key=lambda path: path.name)
    if len(sorted_dirs) == 0:
        msg = f"Unable to find data directories with given glob pattern: {glob_pattern}"
        raise ValueError(msg)
    return sorted_dirs[-1]


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
            event_logger().error(msg)
            raise ValueError(msg)
        built_path /= filename

    return built_path


# Local Variables:
# jinx-local-words: "dT dir noqa sublabel symlink"
# End:
