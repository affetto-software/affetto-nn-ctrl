from __future__ import annotations

import datetime
import itertools
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from affetto_nn_ctrl import DEFAULT_BASE_DIR_PATH, TESTS_DIR_PATH
from affetto_nn_ctrl.data_handling import (
    build_data_dir_path,
    build_data_file_path,
    get_default_base_dir,
    get_default_counter,
    prepare_data_dir_path,
    split_data_dir_path_by_date,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def output_dir_path() -> Path:
    return DEFAULT_BASE_DIR_PATH / "app" / "testing"


@pytest.fixture
def make_work_directory() -> Generator[Path, Any, Any]:
    work_dir = TESTS_DIR_PATH / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    yield work_dir
    shutil.rmtree(work_dir)


def test_get_default_base_dir(make_work_directory: Path) -> None:
    base_dir_config = make_work_directory / "base_dir"
    expected = "/home/user/shared/data/affetto_nn_ctrl"
    text = f" {expected}" + "\n"  # intentionally include white spaces
    base_dir_config.write_text(text, encoding="utf-8")
    default_base_dir = get_default_base_dir(base_dir_config)
    assert str(default_base_dir) == expected


@pytest.mark.parametrize(
    ("data_dir_path", "expected"),
    [
        (Path("/home/user"), (Path("/home/user"), None, None)),
        (Path("/home/user/20240925T130350"), (Path("/home/user"), "20240925T130350", "")),
        (Path("/home/user/20240925T130350.598576"), (Path("/home/user"), "20240925T130350.598576", "")),
        (Path("/home/user/20240925T130352/sub"), (Path("/home/user"), "20240925T130352", "sub")),
        (Path("/home/user/20240925T130352.285998/sub"), (Path("/home/user"), "20240925T130352.285998", "sub")),
        (Path("/home/user/20240925T130355/sub/dir"), (Path("/home/user"), "20240925T130355", "sub/dir")),
        (Path("/home/20240925T130355.532896/sub/dir"), (Path("/home"), "20240925T130355.532896", "sub/dir")),
        (Path("some/where"), (Path("some/where"), None, None)),
        (Path("./data"), (Path("data"), None, None)),
        (Path("./some/where/20240925T130350"), (Path("some/where"), "20240925T130350", "")),
        (Path("some/where/20240925T130350.598576"), (Path("some/where"), "20240925T130350.598576", "")),
        (Path("some/where/20240925T130352/sub"), (Path("some/where"), "20240925T130352", "sub")),
        (Path("./some/where/20240925T130352.285998/sub"), (Path("some/where"), "20240925T130352.285998", "sub")),
        (Path("some/where/20240925T130355/sub/dir"), (Path("some/where"), "20240925T130355", "sub/dir")),
        (Path("./data/20240925T130355.532896/sub/dir"), (Path("data"), "20240925T130355.532896", "sub/dir")),
    ],
)
def test_split_data_dir_path_by_date(data_dir_path: Path, expected: tuple[Path, str | None, str | None]) -> None:
    ret = split_data_dir_path_by_date(data_dir_path)
    assert ret == expected


def test_build_data_dir_path_default_basedir() -> None:
    expected = Path(__file__).parent.parent / "data" / "app" / "test"
    path = build_data_dir_path(base_dir=None, split_by_date=False)
    assert path == expected


@pytest.mark.parametrize("base_dir", [".", "data", "./data", Path.home()])
def test_build_data_dir_path_basedir(base_dir: str | Path) -> None:
    expected = Path(base_dir)
    path = build_data_dir_path(base_dir=base_dir, app_name="", label="", split_by_date=False)
    assert path == expected


@pytest.mark.parametrize(("app_name", "label"), [("app", "test"), ("performance", "testing")])
def test_build_data_dir_path_app_name_and_label(app_name: str, label: str) -> None:
    expected = DEFAULT_BASE_DIR_PATH / app_name / label
    path = build_data_dir_path(base_dir=None, app_name=app_name, label=label, split_by_date=False)
    assert path == expected


@pytest.mark.parametrize(("app_name", "label"), [("app", "test")])
def test_build_data_dir_path_split_by_date(app_name: str, label: str) -> None:
    today = datetime.datetime.now().strftime("%Y%m%d")  # noqa: DTZ005
    expected = DEFAULT_BASE_DIR_PATH / app_name / label / today
    expected_re = re.compile(str(expected) + r"T[0-9]{6}$")
    path = build_data_dir_path(base_dir=None, app_name=app_name, label=label, split_by_date=True)
    assert expected_re.match(str(path)) is not None


@pytest.mark.parametrize(("app_name", "label"), [("app", "test")])
def test_build_data_dir_path_split_by_date_millisecond(app_name: str, label: str) -> None:
    today = datetime.datetime.now().strftime("%Y%m%d")  # noqa: DTZ005
    expected = DEFAULT_BASE_DIR_PATH / app_name / label / today
    expected_re = re.compile(str(expected) + r"T[0-9]{6}\.[0-9]{6}$")
    path = build_data_dir_path(
        base_dir=None,
        app_name=app_name,
        label=label,
        split_by_date=True,
        millisecond=True,
    )
    assert expected_re.match(str(path)) is not None


@pytest.mark.parametrize(
    ("app_name", "label", "sublabel", "split_by_date"),
    [("app", "test", "sublabel_A", False), ("performance", "testing", "sublabel_B", True)],
)
def test_build_data_dir_path_sublabel(app_name: str, label: str, sublabel: str, split_by_date: bool) -> None:  # noqa: FBT001
    today = datetime.datetime.now().strftime("%Y%m%d")  # noqa: DTZ005
    path = build_data_dir_path(
        base_dir=None,
        app_name=app_name,
        label=label,
        sublabel=sublabel,
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
def test_build_data_dir_path_sublabel_zero_length(
    app_name: str,
    label: str,
    sublabel: str,
    split_by_date: bool,  # noqa: FBT001
) -> None:
    today = datetime.datetime.now().strftime("%Y%m%d")  # noqa: DTZ005
    path = build_data_dir_path(
        base_dir=None,
        app_name=app_name,
        label=label,
        sublabel=sublabel,
        split_by_date=split_by_date,
    )
    if not split_by_date:
        expected = DEFAULT_BASE_DIR_PATH / app_name / label
        assert path == expected
    else:
        expected = DEFAULT_BASE_DIR_PATH / app_name / label / today
        expected_re = re.compile(str(expected) + r"T[0-9]{6}$")
        assert expected_re.match(str(path)) is not None


@pytest.mark.parametrize(("label", "date"), [("dataset", "20240925T113437"), ("good_data", "20240925T113441.618707")])
def test_prepare_data_dir_path(make_work_directory: Path, label: str, date: str) -> None:
    data_dir_path = make_work_directory / label / date
    path = prepare_data_dir_path(data_dir_path)
    assert path.exists()
    assert path.is_dir()


@pytest.mark.parametrize(("label", "date"), [("dataset", "20240925T113437"), ("good_data", "20240925T113448.618507")])
def test_prepare_data_dir_path_exists_ok(make_work_directory: Path, label: str, date: str) -> None:
    data_dir_path = make_work_directory / label / date
    data_dir_path.mkdir(parents=True)
    path = prepare_data_dir_path(data_dir_path)
    assert path.exists()
    assert path.is_dir()


@pytest.mark.parametrize(
    ("label", "date", "sublabel"),
    [
        ("dataset", "20240925T120437", "sub_dataset_A"),
        ("good_data", "20240925T113442.721703", "sub_dataset_B"),
        ("bad_data", None, "sub_dataset_C"),
    ],
)
def test_prepare_data_dir_path_with_sublabel(
    make_work_directory: Path,
    label: str,
    date: str | None,
    sublabel: str,
) -> None:
    data_dir_path = make_work_directory / label
    if date is not None:
        data_dir_path /= date
    data_dir_path /= sublabel
    path = prepare_data_dir_path(data_dir_path)
    assert path.exists()
    assert path.is_dir()


@pytest.mark.parametrize(("label", "date"), [("dataset", "20240925T120931"), ("good_data", "20240925T120938.462639")])
def test_prepare_data_dir_path_make_symlink(make_work_directory: Path, label: str, date: str) -> None:
    data_dir_path = make_work_directory / label / date
    symlink_path = make_work_directory / label / "latest"
    prepare_data_dir_path(data_dir_path, make_latest_symlink=True)
    assert data_dir_path.is_dir()
    assert symlink_path.is_dir()
    assert symlink_path.is_symlink()


def test_prepare_data_dir_path_make_symlink_but_exists(make_work_directory: Path) -> None:
    label = "dataset"
    old_data_dir_path = make_work_directory / label / "20240925T121628"
    data_dir_path = make_work_directory / label / "20240925T121845"
    symlink_path = make_work_directory / label / "latest"
    # symbolic link exists already
    old_data_dir_path.mkdir(parents=True)
    os.symlink(old_data_dir_path, symlink_path)
    prepare_data_dir_path(data_dir_path, make_latest_symlink=True)
    assert data_dir_path.is_dir()
    assert symlink_path.is_dir()
    assert symlink_path.is_symlink()


@pytest.mark.parametrize(
    ("label", "date", "sublabel"),
    [
        ("dataset", "20240925T123011", "sub_dataset"),
        ("good_data", "20240925T123022.732009", "excellent"),
    ],
)
def test_prepare_data_dir_path_make_symlink_with_sublabel(
    make_work_directory: Path,
    label: str,
    date: str,
    sublabel: str,
) -> None:
    data_dir_path = make_work_directory / label / date / sublabel
    symlink_path = make_work_directory / label / "latest"
    symlink_dst_path = symlink_path / sublabel
    prepare_data_dir_path(data_dir_path, make_latest_symlink=True)
    assert data_dir_path.is_dir()
    assert symlink_dst_path.is_dir()
    assert symlink_path.is_symlink()


def test_prepare_data_dir_path_make_symlink_but_no_date(make_work_directory: Path) -> None:
    data_dir_path = make_work_directory / "label" / "sublabel"
    with pytest.warns(UserWarning) as record:
        prepare_data_dir_path(data_dir_path, make_latest_symlink=True)
    assert len(record) == 1
    msg = "Trying to make latest symlink, but no date part has found"
    assert str(record[0].message).startswith(msg)


@pytest.mark.parametrize(("prefix", "ext"), [("data", ".csv"), ("image", ".png")])
def test_build_data_file_path(output_dir_path: Path, prefix: str, ext: str) -> None:
    expected = output_dir_path / f"{prefix}{ext}"
    path = build_data_file_path(
        output_dir_path,
        prefix=prefix,
        ext=ext,
    )
    assert path == expected


@pytest.mark.parametrize(("prefix", "ext"), [("data", "csv"), ("image", "png")])
def test_build_data_file_path_ext_no_dot(output_dir_path: Path, prefix: str, ext: str) -> None:
    expected = output_dir_path / f"{prefix}.{ext}"
    path = build_data_file_path(
        output_dir_path,
        prefix=prefix,
        ext=ext,
    )
    assert path == expected


@pytest.mark.parametrize(("prefix", "ext"), [("data", ".csv"), ("", ".png")])
def test_build_data_file_path_ext_with_iterator(output_dir_path: Path, prefix: str, ext: str) -> None:
    cnt = itertools.count()
    # 0
    path = build_data_file_path(
        output_dir_path,
        prefix=prefix,
        iterator=cnt,
        ext=ext,
    )
    expected = output_dir_path / f"{prefix}0{ext}"
    assert path == expected

    # 1
    path = build_data_file_path(
        output_dir_path,
        prefix=prefix,
        iterator=cnt,
        ext=ext,
    )
    expected = output_dir_path / f"{prefix}1{ext}"
    assert path == expected


@pytest.mark.parametrize(("prefix", "ext"), [("data", ".csv"), ("", ".png")])
def test_build_data_file_path_ext_with_default_counter(output_dir_path: Path, prefix: str, ext: str) -> None:
    cnt = get_default_counter()
    # 0
    path = build_data_file_path(
        output_dir_path,
        prefix=prefix,
        iterator=cnt,
        ext=ext,
    )
    expected = output_dir_path / f"{prefix}000{ext}"
    assert path == expected

    # 1
    path = build_data_file_path(
        output_dir_path,
        prefix=prefix,
        iterator=cnt,
        ext=ext,
    )
    expected = output_dir_path / f"{prefix}001{ext}"
    assert path == expected


def test_build_data_file_path_ext_raise_error_when_no_prefix_iterator(output_dir_path: Path) -> None:
    msg = "Extension was given, but unable to determine filename"
    with pytest.raises(ValueError, match=msg):
        _ = build_data_file_path(
            output_dir_path,
            prefix="",
            iterator=None,
            ext=".csv",
        )


@pytest.mark.parametrize(("prefix"), [("directory"), ("dir_")])
def test_build_data_file_path_when_ext_is_zero_make_directory(output_dir_path: Path, prefix: str) -> None:
    cnt = get_default_counter()
    # 0
    path = build_data_file_path(
        output_dir_path,
        prefix=prefix,
        iterator=cnt,
        ext="",
    )
    expected = output_dir_path / f"{prefix}000"
    assert path == expected

    # 1
    path = build_data_file_path(
        output_dir_path,
        prefix=prefix,
        iterator=cnt,
        ext="",
    )
    expected = output_dir_path / f"{prefix}001"
    assert path == expected


# Local Variables:
# jinx-local-words: "csv ctrl dataset dir nn noqa png sublabel symlink"
# End:
