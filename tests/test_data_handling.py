from __future__ import annotations

import datetime
import itertools
import re
from pathlib import Path

import pytest

from affetto_nn_ctrl import DEFAULT_BASE_DIR_PATH
from affetto_nn_ctrl.data_handling import get_data_path, get_default_counter


def test_get_filename_default_basedir() -> None:
    expected = Path(__file__).parent.parent / "data" / "app" / "test"
    path = get_data_path(base_dir=None, ext=None, split_by_date=False)
    assert path == expected


@pytest.mark.parametrize("base_dir", [".", "data", "./data", Path.home()])
def test_get_filename_basedir(base_dir: str | Path) -> None:
    expected = Path(base_dir)
    path = get_data_path(base_dir=base_dir, app_name="", label="", ext=None, split_by_date=False)
    assert path == expected


@pytest.mark.parametrize(("app_name", "label"), [("app", "test"), ("performance", "testing")])
def test_get_filename_app_name_and_label(app_name: str, label: str) -> None:
    expected = DEFAULT_BASE_DIR_PATH / app_name / label
    path = get_data_path(base_dir=None, app_name=app_name, label=label, ext=None, split_by_date=False)
    assert path == expected


@pytest.mark.parametrize(("app_name", "label"), [("app", "test")])
def test_get_filename_split_by_date(app_name: str, label: str) -> None:
    today = datetime.datetime.now().strftime("%Y%m%d")  # noqa: DTZ005
    expected = DEFAULT_BASE_DIR_PATH / app_name / label / today
    expected_re = re.compile(str(expected) + r"T[0-9]{6}$")
    path = get_data_path(base_dir=None, app_name=app_name, label=label, ext=None, split_by_date=True)
    assert expected_re.match(str(path)) is not None


@pytest.mark.parametrize(("app_name", "label"), [("app", "test")])
def test_get_filename_split_by_date_millisecond(app_name: str, label: str) -> None:
    today = datetime.datetime.now().strftime("%Y%m%d")  # noqa: DTZ005
    expected = DEFAULT_BASE_DIR_PATH / app_name / label / today
    expected_re = re.compile(str(expected) + r"T[0-9]{6}\.[0-9]{6}$")
    path = get_data_path(
        base_dir=None,
        app_name=app_name,
        label=label,
        ext=None,
        split_by_date=True,
        millisecond=True,
    )
    assert expected_re.match(str(path)) is not None


@pytest.mark.parametrize(
    ("app_name", "label", "sublabel", "split_by_date"),
    [("app", "test", "sublabel_A", False), ("performance", "testing", "sublabel_B", True)],
)
def test_get_filename_sublabel(app_name: str, label: str, sublabel: str, split_by_date: bool) -> None:  # noqa: FBT001
    today = datetime.datetime.now().strftime("%Y%m%d")  # noqa: DTZ005
    path = get_data_path(
        base_dir=None,
        app_name=app_name,
        label=label,
        sublabel=sublabel,
        ext=None,
        split_by_date=split_by_date,
    )
    if not split_by_date:
        expected = DEFAULT_BASE_DIR_PATH / app_name / label / sublabel
        assert path == expected
    else:
        expected = DEFAULT_BASE_DIR_PATH / app_name / label / today
        expected_re = re.compile(str(expected) + r"T[0-9]{6}/" + sublabel)
        assert expected_re.match(str(path)) is not None


@pytest.mark.parametrize(
    ("app_name", "label", "sublabel", "split_by_date"),
    [("app", "test", "", False), ("performance", "testing", "", True)],
)
def test_get_filename_sublabel_zero_length(app_name: str, label: str, sublabel: str, split_by_date: bool) -> None:  # noqa: FBT001
    today = datetime.datetime.now().strftime("%Y%m%d")  # noqa: DTZ005
    path = get_data_path(
        base_dir=None,
        app_name=app_name,
        label=label,
        sublabel=sublabel,
        ext=None,
        split_by_date=split_by_date,
    )
    if not split_by_date:
        expected = DEFAULT_BASE_DIR_PATH / app_name / label
        assert path == expected
    else:
        expected = DEFAULT_BASE_DIR_PATH / app_name / label / today
        expected_re = re.compile(str(expected) + r"T[0-9]{6}$")
        assert expected_re.match(str(path)) is not None


@pytest.mark.parametrize(
    ("app_name", "label", "prefix", "ext"),
    [("app", "test", "data", ".csv"), ("performance", "testing", "image", ".png")],
)
def test_get_filename_ext(app_name: str, label: str, prefix: str, ext: str) -> None:
    expected = DEFAULT_BASE_DIR_PATH / app_name / label / f"{prefix}{ext}"
    path = get_data_path(
        base_dir=None,
        app_name=app_name,
        label=label,
        prefix=prefix,
        ext=ext,
        split_by_date=False,
    )
    assert path == expected


@pytest.mark.parametrize(
    ("app_name", "label", "prefix", "ext"),
    [("app", "test", "data", "csv"), ("performance", "testing", "image", "png")],
)
def test_get_filename_ext_no_dot(app_name: str, label: str, prefix: str, ext: str) -> None:
    expected = DEFAULT_BASE_DIR_PATH / app_name / label / f"{prefix}.{ext}"
    path = get_data_path(
        base_dir=None,
        app_name=app_name,
        label=label,
        prefix=prefix,
        ext=ext,
        split_by_date=False,
    )
    assert path == expected


@pytest.mark.parametrize(
    ("app_name", "label", "prefix", "ext"),
    [("app", "test", "data", ".csv"), ("performance", "testing", "", ".png")],
)
def test_get_filename_ext_with_iterator(app_name: str, label: str, prefix: str, ext: str) -> None:
    cnt = itertools.count()
    # 0
    path = get_data_path(
        base_dir=None,
        app_name=app_name,
        label=label,
        prefix=prefix,
        iterator=cnt,
        ext=ext,
        split_by_date=False,
    )
    expected = DEFAULT_BASE_DIR_PATH / app_name / label / f"{prefix}0{ext}"
    assert path == expected

    # 1
    path = get_data_path(
        base_dir=None,
        app_name=app_name,
        label=label,
        prefix=prefix,
        iterator=cnt,
        ext=ext,
        split_by_date=False,
    )
    expected = DEFAULT_BASE_DIR_PATH / app_name / label / f"{prefix}1{ext}"
    assert path == expected


@pytest.mark.parametrize(
    ("app_name", "label", "prefix", "ext"),
    [("app", "test", "data", ".csv"), ("performance", "testing", "", ".png")],
)
def test_get_filename_ext_with_default_counter(app_name: str, label: str, prefix: str, ext: str) -> None:
    cnt = get_default_counter()
    # 0
    path = get_data_path(
        base_dir=None,
        app_name=app_name,
        label=label,
        prefix=prefix,
        iterator=cnt,
        ext=ext,
        split_by_date=False,
    )
    expected = DEFAULT_BASE_DIR_PATH / app_name / label / f"{prefix}000{ext}"
    assert path == expected

    # 1
    path = get_data_path(
        base_dir=None,
        app_name=app_name,
        label=label,
        prefix=prefix,
        iterator=cnt,
        ext=ext,
        split_by_date=False,
    )
    expected = DEFAULT_BASE_DIR_PATH / app_name / label / f"{prefix}001{ext}"
    assert path == expected


def test_get_filename_ext_raise_error_when_no_prefix_iterator() -> None:
    msg = "Extension was given, but unable to determine filename"
    with pytest.raises(ValueError, match=msg):
        _ = get_data_path(
            base_dir=None,
            prefix="",
            iterator=None,
            ext=".csv",
        )


@pytest.mark.parametrize(
    ("app_name", "label", "sublabel", "prefix"),
    [("app", "test", "subset_A", "directory"), ("app", "test", "subset_B", "dir_")],
)
def test_get_filename_when_ext_is_zero_make_directory(app_name: str, label: str, sublabel: str, prefix: str) -> None:
    cnt = get_default_counter()
    # 0
    path = get_data_path(
        base_dir=None,
        app_name=app_name,
        label=label,
        sublabel=sublabel,
        prefix=prefix,
        iterator=cnt,
        ext="",
        split_by_date=False,
    )
    expected = DEFAULT_BASE_DIR_PATH / app_name / label / sublabel / f"{prefix}000"
    assert path == expected

    # 1
    path = get_data_path(
        base_dir=None,
        app_name=app_name,
        label=label,
        sublabel=sublabel,
        prefix=prefix,
        iterator=cnt,
        ext="",
        split_by_date=False,
    )
    expected = DEFAULT_BASE_DIR_PATH / app_name / label / sublabel / f"{prefix}001"
    assert path == expected


# Local Variables:
# jinx-local-words: "csv dir noqa png sublabel"
# End:
