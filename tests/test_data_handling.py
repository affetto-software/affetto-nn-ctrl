from __future__ import annotations

import datetime
import itertools
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import pytest

from affetto_nn_ctrl import DEFAULT_BASE_DIR_PATH, TESTS_DIR_PATH
from affetto_nn_ctrl.data_handling import (
    build_data_dir_path,
    build_data_file_path,
    find_latest_data_dir_path,
    get_default_base_dir,
    get_default_counter,
    prepare_data_dir_path,
    split_data_dir_path_by_date,
    train_test_split_files,
)
from tests import TESTS_DATA_DIR_PATH

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


class DateDirMaker(Protocol):
    def __call__(
        self,
        base_dir: Path,
        app_name: str,
        label: str,
        *,
        make_latest_symlink: bool = True,
    ) -> Path: ...


def make_date_directories(
    base_dir: Path,
    app_name: str,
    label: str,
    *,
    make_latest_symlink: bool = True,
) -> Path:
    labeled_dir = base_dir / app_name / label
    random_date_strings = [
        "20241024T144633",
        "20241024T145114",
        "20241025T021116",
        "20241025T021232",
    ]
    for date in random_date_strings:
        (labeled_dir / date).mkdir(parents=True, exist_ok=True)
    if make_latest_symlink:
        target = labeled_dir / sorted(random_date_strings)[-1]
        link = labeled_dir / "latest"
        if link.exists():
            link.unlink()
        os.symlink(target.absolute(), link)
    return labeled_dir


def make_date_directories_millisecond(
    base_dir: Path,
    app_name: str,
    label: str,
    *,
    make_latest_symlink: bool = True,
) -> Path:
    labeled_dir = base_dir / app_name / label
    random_date_strings = [
        "20241024T144633.942693",
        "20241024T145114.311196",
        "20241025T021116.011509",
        "20241025T021232.467673",
        "20241025T021232.527792",
    ]
    for date in random_date_strings:
        (labeled_dir / date).mkdir(parents=True, exist_ok=True)
    if make_latest_symlink:
        target = labeled_dir / sorted(random_date_strings)[-1]
        link = labeled_dir / "latest"
        if link.exists():
            link.unlink()
        os.symlink(target.absolute(), link)
    return labeled_dir


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


@pytest.mark.parametrize(
    ("app_name", "label", "sublabel", "specified_date"),
    [
        ("app", "test", None, "20241024T164152"),
        ("performance", "testing", "sub", "20241024T164152.564412"),
    ],
)
def test_build_data_dir_path_specify_date_string(
    app_name: str,
    label: str,
    sublabel: str | None,
    specified_date: str,
) -> None:
    path = build_data_dir_path(
        base_dir=None,
        app_name=app_name,
        label=label,
        sublabel=sublabel,
        specified_date=specified_date,
    )
    if sublabel is not None:
        expected = DEFAULT_BASE_DIR_PATH / app_name / label / specified_date / sublabel
    else:
        expected = DEFAULT_BASE_DIR_PATH / app_name / label / specified_date
    assert path == expected


@pytest.mark.parametrize(
    ("app_name", "label", "sublabel", "date_dir_maker", "expected_date"),
    [
        ("app", "test", None, make_date_directories, "20241025T021232"),
        ("performance", "testing", "sub", make_date_directories_millisecond, "20241025T021232.527792"),
    ],
)
def test_build_data_dir_path_specify_latest(
    make_work_directory: Path,
    app_name: str,
    label: str,
    sublabel: str | None,
    date_dir_maker: DateDirMaker,
    expected_date: str,
) -> None:
    base_dir = make_work_directory
    labeled_data_dir_path = date_dir_maker(base_dir, app_name, label)
    path = build_data_dir_path(
        base_dir=base_dir,
        app_name=app_name,
        label=label,
        sublabel=sublabel,
        specified_date="latest",
    )
    if sublabel is not None:
        expected = labeled_data_dir_path / expected_date / sublabel
    else:
        expected = labeled_data_dir_path / expected_date
    assert path == expected


@pytest.mark.parametrize(
    ("app_name", "label", "date_dir_maker", "expected_date"),
    [
        ("dataset", "find_latest", make_date_directories, "20241025T021232"),
        ("dataset", "find_latest", make_date_directories_millisecond, "20241025T021232.527792"),
    ],
)
def test_find_latest_data_dir_path(
    make_work_directory: Path,
    app_name: str,
    label: str,
    date_dir_maker: DateDirMaker,
    expected_date: str,
) -> None:
    base_dir = make_work_directory
    labeled_data_dir_path = date_dir_maker(base_dir, app_name, label)
    expected_latest_dir_path = labeled_data_dir_path / expected_date
    found_dir_path = find_latest_data_dir_path(base_dir, app_name, label)
    assert found_dir_path == expected_latest_dir_path


@pytest.mark.parametrize(
    ("app_name", "label", "date_dir_maker", "expected_date"),
    [
        ("dataset", "find_latest", make_date_directories, "20241025T021232"),
        ("dataset", "find_latest", make_date_directories_millisecond, "20241025T021232.527792"),
    ],
)
def test_find_latest_data_dir_path_specify_search_dir(
    make_work_directory: Path,
    app_name: str,
    label: str,
    date_dir_maker: DateDirMaker,
    expected_date: str,
) -> None:
    base_dir = make_work_directory
    search_dir = base_dir / app_name / label
    labeled_data_dir_path = date_dir_maker(base_dir, app_name, label)
    expected_latest_dir_path = labeled_data_dir_path / expected_date
    found_dir_path = find_latest_data_dir_path(search_dir)
    assert found_dir_path == expected_latest_dir_path


@pytest.mark.parametrize(
    ("app_name", "label", "date_dir_maker", "expected_date"),
    [
        ("dataset", "find_latest", make_date_directories, "20241025T021232"),
        ("dataset", "find_latest", make_date_directories_millisecond, "20241025T021232.527792"),
    ],
)
def test_find_latest_data_dir_path_link_not_exist(
    make_work_directory: Path,
    app_name: str,
    label: str,
    date_dir_maker: DateDirMaker,
    expected_date: str,
) -> None:
    base_dir = make_work_directory
    labeled_data_dir_path = date_dir_maker(base_dir, app_name, label, make_latest_symlink=False)
    expected_latest_dir_path = labeled_data_dir_path / expected_date
    found_dir_path = find_latest_data_dir_path(base_dir, app_name, label)
    assert found_dir_path == expected_latest_dir_path


@pytest.mark.parametrize(
    ("app_name", "label", "date_dir_maker", "expected_date"),
    [
        ("dataset", "find_latest", make_date_directories, "20241025T021232"),
        ("dataset", "find_latest", make_date_directories_millisecond, "20241025T021232.527792"),
    ],
)
def test_find_latest_data_dir_path_link_exists_but_no_target(
    make_work_directory: Path,
    app_name: str,
    label: str,
    date_dir_maker: DateDirMaker,
    expected_date: str,
) -> None:
    base_dir = make_work_directory
    app_name = "dataset"
    label = "find_latest"
    labeled_data_dir_path = date_dir_maker(base_dir, app_name, label, make_latest_symlink=False)
    link = labeled_data_dir_path / "latest"
    os.symlink("/not/exist/directory", link)
    expected_latest_dir_path = labeled_data_dir_path / expected_date
    found_dir_path = find_latest_data_dir_path(base_dir, app_name, label)
    assert found_dir_path == expected_latest_dir_path


@pytest.mark.parametrize(
    ("app_name", "label", "date_dir_maker", "expected_date", "fake_date"),
    [
        ("dataset", "find_latest", make_date_directories, "20241025T021232", "20241025T021116"),
        (
            "dataset",
            "find_latest",
            make_date_directories_millisecond,
            "20241025T021232.527792",
            "20241025T021232.467673",
        ),
    ],
)
def test_find_latest_data_dir_path_force_find_by_pattern(
    make_work_directory: Path,
    app_name: str,
    label: str,
    date_dir_maker: DateDirMaker,
    expected_date: str,
    fake_date: str,
) -> None:
    base_dir = make_work_directory
    app_name = "dataset"
    label = "find_latest"
    labeled_data_dir_path = date_dir_maker(base_dir, app_name, label, make_latest_symlink=False)
    link = labeled_data_dir_path / "latest"
    # intentionally make symlink to not latest directory
    target = labeled_data_dir_path / fake_date
    os.symlink(target, link)
    expected_latest_dir_path = labeled_data_dir_path / expected_date
    found_dir_path = find_latest_data_dir_path(base_dir, app_name, label, force_find_by_pattern=True)
    assert found_dir_path == expected_latest_dir_path


@pytest.mark.parametrize(
    ("app_name", "label", "date_dir_maker"),
    [
        ("dataset", "find_latest", make_date_directories),
        ("dataset", "find_latest", make_date_directories_millisecond),
    ],
)
def test_find_latest_data_dir_path_error_invalid_pattern(
    make_work_directory: Path,
    app_name: str,
    label: str,
    date_dir_maker: DateDirMaker,
) -> None:
    base_dir = make_work_directory
    app_name = "dataset"
    label = "find_latest"
    date_dir_maker(base_dir, app_name, label)
    with pytest.raises(ValueError, match="Unable to find data directories with given glob pattern"):
        find_latest_data_dir_path(base_dir, app_name, label, glob_pattern="1970*T*", force_find_by_pattern=True)


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
    expected = output_dir_path / f"{prefix}_000{ext}"
    assert path == expected

    # 1
    path = build_data_file_path(
        output_dir_path,
        prefix=prefix,
        iterator=cnt,
        ext=ext,
    )
    expected = output_dir_path / f"{prefix}_001{ext}"
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


@pytest.mark.parametrize(("prefix"), [("directory"), ("dir")])
def test_build_data_file_path_when_ext_is_zero_make_directory(output_dir_path: Path, prefix: str) -> None:
    cnt = get_default_counter()
    # 0
    path = build_data_file_path(
        output_dir_path,
        prefix=prefix,
        iterator=cnt,
        ext="",
    )
    expected = output_dir_path / f"{prefix}_000"
    assert path == expected

    # 1
    path = build_data_file_path(
        output_dir_path,
        prefix=prefix,
        iterator=cnt,
        ext="",
    )
    expected = output_dir_path / f"{prefix}_001"
    assert path == expected


@pytest.mark.parametrize(
    ("dataset_path", "test_size", "train_size", "glob_pattern", "expected"),
    [
        (
            "dummy_datasets_01",
            None,
            None,
            "*.csv",
            (
                [
                    "dummy_data_000.csv",
                    "dummy_data_002.csv",
                    "dummy_data_001.csv",
                    "dummy_data_005.csv",
                    "dummy_data_006.csv",
                    "dummy_data_008.csv",
                    "dummy_data_003.csv",
                    "dummy_data_009.csv",
                ],
                [
                    "dummy_data_007.csv",
                    "dummy_data_004.csv",
                ],
            ),
        ),
        (
            "dummy_datasets_01",
            0.33,
            None,
            "*.csv",
            (
                [
                    "dummy_data_002.csv",
                    "dummy_data_001.csv",
                    "dummy_data_005.csv",
                    "dummy_data_006.csv",
                    "dummy_data_008.csv",
                    "dummy_data_003.csv",
                    "dummy_data_009.csv",
                ],
                [
                    "dummy_data_007.csv",
                    "dummy_data_004.csv",
                    "dummy_data_000.csv",
                ],
            ),
        ),
        (
            "dummy_datasets_01",
            None,
            0.9,
            "*.csv",
            (
                [
                    "dummy_data_004.csv",
                    "dummy_data_000.csv",
                    "dummy_data_002.csv",
                    "dummy_data_001.csv",
                    "dummy_data_005.csv",
                    "dummy_data_006.csv",
                    "dummy_data_008.csv",
                    "dummy_data_003.csv",
                    "dummy_data_009.csv",
                ],
                [
                    "dummy_data_007.csv",
                ],
            ),
        ),
        (
            "dummy_datasets_01",
            0.4,
            0.6,
            "*.csv",
            (
                [
                    "dummy_data_001.csv",
                    "dummy_data_005.csv",
                    "dummy_data_006.csv",
                    "dummy_data_008.csv",
                    "dummy_data_003.csv",
                    "dummy_data_009.csv",
                ],
                [
                    "dummy_data_007.csv",
                    "dummy_data_004.csv",
                    "dummy_data_000.csv",
                    "dummy_data_002.csv",
                ],
            ),
        ),
        (
            "dummy_datasets_01",
            0.3,
            0.5,
            "*.csv",
            (
                [
                    "dummy_data_002.csv",
                    "dummy_data_001.csv",
                    "dummy_data_005.csv",
                    "dummy_data_006.csv",
                    "dummy_data_008.csv",
                ],
                [
                    "dummy_data_007.csv",
                    "dummy_data_004.csv",
                    "dummy_data_000.csv",
                ],
            ),
        ),
        (
            "dummy_datasets_01",
            0.0,
            None,
            "*.csv",
            (
                [
                    "dummy_data_007.csv",
                    "dummy_data_004.csv",
                    "dummy_data_000.csv",
                    "dummy_data_002.csv",
                    "dummy_data_001.csv",
                    "dummy_data_005.csv",
                    "dummy_data_006.csv",
                    "dummy_data_008.csv",
                    "dummy_data_003.csv",
                    "dummy_data_009.csv",
                ],
                [],
            ),
        ),
    ],
)
def test_train_test_split_files_ratio(
    dataset_path: str,
    test_size: float | None,
    train_size: float | None,
    glob_pattern: str,
    expected: tuple[list[str], list[str]],
) -> None:
    seed = 123
    dirpath = TESTS_DATA_DIR_PATH / dataset_path
    train, test = train_test_split_files(dirpath, test_size, train_size, glob_pattern, seed)
    assert train == [Path(dirpath / x) for x in expected[0]]
    assert test == [Path(dirpath / x) for x in expected[1]]


@pytest.mark.parametrize(
    ("dataset_path", "test_size", "train_size", "glob_pattern", "expected"),
    [
        (
            "dummy_datasets_01",
            None,
            None,
            "*.csv",
            (
                [
                    "dummy_data_000.csv",
                    "dummy_data_002.csv",
                    "dummy_data_001.csv",
                    "dummy_data_005.csv",
                    "dummy_data_006.csv",
                    "dummy_data_008.csv",
                    "dummy_data_003.csv",
                    "dummy_data_009.csv",
                ],
                [
                    "dummy_data_007.csv",
                    "dummy_data_004.csv",
                ],
            ),
        ),
        (
            "dummy_datasets_01",
            3,
            None,
            "*.csv",
            (
                [
                    "dummy_data_002.csv",
                    "dummy_data_001.csv",
                    "dummy_data_005.csv",
                    "dummy_data_006.csv",
                    "dummy_data_008.csv",
                    "dummy_data_003.csv",
                    "dummy_data_009.csv",
                ],
                [
                    "dummy_data_007.csv",
                    "dummy_data_004.csv",
                    "dummy_data_000.csv",
                ],
            ),
        ),
        (
            "dummy_datasets_01",
            None,
            9,
            "*.csv",
            (
                [
                    "dummy_data_004.csv",
                    "dummy_data_000.csv",
                    "dummy_data_002.csv",
                    "dummy_data_001.csv",
                    "dummy_data_005.csv",
                    "dummy_data_006.csv",
                    "dummy_data_008.csv",
                    "dummy_data_003.csv",
                    "dummy_data_009.csv",
                ],
                [
                    "dummy_data_007.csv",
                ],
            ),
        ),
        (
            "dummy_datasets_01",
            4,
            6,
            "*.csv",
            (
                [
                    "dummy_data_001.csv",
                    "dummy_data_005.csv",
                    "dummy_data_006.csv",
                    "dummy_data_008.csv",
                    "dummy_data_003.csv",
                    "dummy_data_009.csv",
                ],
                [
                    "dummy_data_007.csv",
                    "dummy_data_004.csv",
                    "dummy_data_000.csv",
                    "dummy_data_002.csv",
                ],
            ),
        ),
        (
            "dummy_datasets_01",
            3,
            5,
            "*.csv",
            (
                [
                    "dummy_data_002.csv",
                    "dummy_data_001.csv",
                    "dummy_data_005.csv",
                    "dummy_data_006.csv",
                    "dummy_data_008.csv",
                ],
                [
                    "dummy_data_007.csv",
                    "dummy_data_004.csv",
                    "dummy_data_000.csv",
                ],
            ),
        ),
        (
            "dummy_datasets_01",
            0,
            None,
            "*.csv",
            (
                [
                    "dummy_data_007.csv",
                    "dummy_data_004.csv",
                    "dummy_data_000.csv",
                    "dummy_data_002.csv",
                    "dummy_data_001.csv",
                    "dummy_data_005.csv",
                    "dummy_data_006.csv",
                    "dummy_data_008.csv",
                    "dummy_data_003.csv",
                    "dummy_data_009.csv",
                ],
                [],
            ),
        ),
    ],
)
def test_train_test_split_files_num(
    dataset_path: str,
    test_size: float | None,
    train_size: float | None,
    glob_pattern: str,
    expected: tuple[list[str], list[str]],
) -> None:
    seed = 123
    dirpath = TESTS_DATA_DIR_PATH / dataset_path
    train, test = train_test_split_files(dirpath, test_size, train_size, glob_pattern, seed)
    assert train == [Path(dirpath / x) for x in expected[0]]
    assert test == [Path(dirpath / x) for x in expected[1]]


@pytest.mark.parametrize(
    ("dataset_path", "test_size", "train_size", "glob_pattern"),
    [
        ("dummy_datasets_01", 0.4, 0.7, "*.csv"),
        ("dummy_datasets_01", 2, 9, "*.csv"),
        ("dummy_datasets_01", None, 1.1, "*.csv"),
        ("dummy_datasets_01", 12, None, "*.csv"),
        ("dummy_datasets_01", 2, 1.2, "*.csv"),
        ("dummy_datasets_01", 12, 0.1, "*.csv"),
    ],
)
def test_train_test_split_files_error(
    dataset_path: str,
    test_size: float | None,
    train_size: float | None,
    glob_pattern: str,
) -> None:
    seed = 123
    dirpath = TESTS_DATA_DIR_PATH / dataset_path
    msg = "Train-test split cannot be done because test/train size exceeds total: "
    with pytest.raises(ValueError, match=msg):
        train_test_split_files(dirpath, test_size, train_size, glob_pattern, seed)


@pytest.mark.parametrize(
    ("dataset_paths", "test_size", "train_size", "glob_pattern", "expected"),
    [
        (
            ["dummy_datasets_01", "dummy_datasets_02"],
            None,
            None,
            "*.csv",
            (
                [
                    "dummy_datasets_02/dummy_data_014.csv",
                    "dummy_datasets_02/dummy_data_000.csv",
                    "dummy_datasets_01/dummy_data_006.csv",
                    "dummy_datasets_02/dummy_data_018.csv",
                    "dummy_datasets_01/dummy_data_001.csv",
                    "dummy_datasets_02/dummy_data_005.csv",
                    "dummy_datasets_01/dummy_data_000.csv",
                    "dummy_datasets_01/dummy_data_008.csv",
                    "dummy_datasets_02/dummy_data_011.csv",
                    "dummy_datasets_02/dummy_data_013.csv",
                    "dummy_datasets_02/dummy_data_010.csv",
                    "dummy_datasets_02/dummy_data_006.csv",
                    "dummy_datasets_02/dummy_data_009.csv",
                    "dummy_datasets_01/dummy_data_009.csv",
                    "dummy_datasets_02/dummy_data_002.csv",
                    "dummy_datasets_02/dummy_data_016.csv",
                    "dummy_datasets_02/dummy_data_007.csv",
                    "dummy_datasets_01/dummy_data_005.csv",
                    "dummy_datasets_02/dummy_data_004.csv",
                    "dummy_datasets_02/dummy_data_012.csv",
                    "dummy_datasets_01/dummy_data_003.csv",
                    "dummy_datasets_02/dummy_data_015.csv",
                    "dummy_datasets_02/dummy_data_003.csv",
                ],
                [
                    "dummy_datasets_01/dummy_data_002.csv",
                    "dummy_datasets_02/dummy_data_019.csv",
                    "dummy_datasets_02/dummy_data_008.csv",
                    "dummy_datasets_01/dummy_data_007.csv",
                    "dummy_datasets_02/dummy_data_017.csv",
                    "dummy_datasets_02/dummy_data_001.csv",
                    "dummy_datasets_01/dummy_data_004.csv",
                ],
            ),
        ),
    ],
)
def test_train_test_split_files_multi_directory(
    dataset_paths: list[str],
    test_size: float | None,
    train_size: float | None,
    glob_pattern: str,
    expected: tuple[list[str], list[str]],
) -> None:
    seed = 123
    dirpaths = [TESTS_DATA_DIR_PATH / x for x in dataset_paths]
    train, test = train_test_split_files(dirpaths, test_size, train_size, glob_pattern, seed)
    assert train == [Path(TESTS_DATA_DIR_PATH / x) for x in expected[0]]
    assert test == [Path(TESTS_DATA_DIR_PATH / x) for x in expected[1]]


@pytest.mark.parametrize(
    ("dataset_paths", "test_size", "train_size", "glob_pattern"),
    [
        (["dummy_datasets_01", "dummy_datasets_02"], None, None, "*.csv"),
    ],
)
def test_train_test_split_files_seed(
    dataset_paths: list[str],
    test_size: float | None,
    train_size: float | None,
    glob_pattern: str,
) -> None:
    seed = 123
    dirpaths = [TESTS_DATA_DIR_PATH / x for x in dataset_paths]
    train01, test01 = train_test_split_files(dirpaths, test_size, train_size, glob_pattern, seed)
    dirpaths = [TESTS_DATA_DIR_PATH / x for x in dataset_paths]
    train02, test02 = train_test_split_files(dirpaths, test_size, train_size, glob_pattern, seed)
    assert train01 == train02
    assert test01 == test02
    seed = 42
    dirpaths = [TESTS_DATA_DIR_PATH / x for x in dataset_paths]
    train03, test03 = train_test_split_files(dirpaths, test_size, train_size, glob_pattern, seed)
    assert train01 != train03
    assert test01 != test03


@pytest.mark.parametrize(
    ("dataset_paths", "test_size", "train_size", "glob_pattern", "split_in_each_directory", "expected"),
    [
        (
            ["dummy_datasets_01", "dummy_datasets_02"],
            None,
            None,
            "*.csv",
            False,
            (
                [
                    "dummy_datasets_02/dummy_data_014.csv",
                    "dummy_datasets_02/dummy_data_000.csv",
                    "dummy_datasets_01/dummy_data_006.csv",
                    "dummy_datasets_02/dummy_data_018.csv",
                    "dummy_datasets_01/dummy_data_001.csv",
                    "dummy_datasets_02/dummy_data_005.csv",
                    "dummy_datasets_01/dummy_data_000.csv",
                    "dummy_datasets_01/dummy_data_008.csv",
                    "dummy_datasets_02/dummy_data_011.csv",
                    "dummy_datasets_02/dummy_data_013.csv",
                    "dummy_datasets_02/dummy_data_010.csv",
                    "dummy_datasets_02/dummy_data_006.csv",
                    "dummy_datasets_02/dummy_data_009.csv",
                    "dummy_datasets_01/dummy_data_009.csv",
                    "dummy_datasets_02/dummy_data_002.csv",
                    "dummy_datasets_02/dummy_data_016.csv",
                    "dummy_datasets_02/dummy_data_007.csv",
                    "dummy_datasets_01/dummy_data_005.csv",
                    "dummy_datasets_02/dummy_data_004.csv",
                    "dummy_datasets_02/dummy_data_012.csv",
                    "dummy_datasets_01/dummy_data_003.csv",
                    "dummy_datasets_02/dummy_data_015.csv",
                    "dummy_datasets_02/dummy_data_003.csv",
                ],
                [
                    "dummy_datasets_01/dummy_data_002.csv",
                    "dummy_datasets_02/dummy_data_019.csv",
                    "dummy_datasets_02/dummy_data_008.csv",
                    "dummy_datasets_01/dummy_data_007.csv",
                    "dummy_datasets_02/dummy_data_017.csv",
                    "dummy_datasets_02/dummy_data_001.csv",
                    "dummy_datasets_01/dummy_data_004.csv",
                ],
            ),
        ),
        (
            ["dummy_datasets_01", "dummy_datasets_02"],
            None,
            None,
            "*.csv",
            True,
            (
                [
                    "dummy_datasets_01/dummy_data_000.csv",
                    "dummy_datasets_01/dummy_data_002.csv",
                    "dummy_datasets_01/dummy_data_001.csv",
                    "dummy_datasets_01/dummy_data_005.csv",
                    "dummy_datasets_01/dummy_data_006.csv",
                    "dummy_datasets_01/dummy_data_008.csv",
                    "dummy_datasets_01/dummy_data_003.csv",
                    "dummy_datasets_01/dummy_data_009.csv",
                    "dummy_datasets_02/dummy_data_011.csv",
                    "dummy_datasets_02/dummy_data_004.csv",
                    "dummy_datasets_02/dummy_data_005.csv",
                    "dummy_datasets_02/dummy_data_010.csv",
                    "dummy_datasets_02/dummy_data_006.csv",
                    "dummy_datasets_02/dummy_data_014.csv",
                    "dummy_datasets_02/dummy_data_001.csv",
                    "dummy_datasets_02/dummy_data_015.csv",
                    "dummy_datasets_02/dummy_data_000.csv",
                    "dummy_datasets_02/dummy_data_008.csv",
                    "dummy_datasets_02/dummy_data_012.csv",
                    "dummy_datasets_02/dummy_data_017.csv",
                    "dummy_datasets_02/dummy_data_009.csv",
                    "dummy_datasets_02/dummy_data_016.csv",
                    "dummy_datasets_02/dummy_data_019.csv",
                ],
                [
                    "dummy_datasets_01/dummy_data_007.csv",
                    "dummy_datasets_01/dummy_data_004.csv",
                    "dummy_datasets_02/dummy_data_002.csv",
                    "dummy_datasets_02/dummy_data_013.csv",
                    "dummy_datasets_02/dummy_data_018.csv",
                    "dummy_datasets_02/dummy_data_007.csv",
                    "dummy_datasets_02/dummy_data_003.csv",
                ],
            ),
        ),
    ],
)
def test_train_test_split_files_split_in_each_directory(
    dataset_paths: list[str],
    test_size: float | None,
    train_size: float | None,
    glob_pattern: str,
    *,
    split_in_each_directory: bool,
    expected: tuple[list[str], list[str]],
) -> None:
    seed = 123
    dirpaths = [TESTS_DATA_DIR_PATH / x for x in dataset_paths]
    train, test = train_test_split_files(
        dirpaths,
        test_size,
        train_size,
        glob_pattern,
        seed,
        split_in_each_directory=split_in_each_directory,
    )
    assert train == [Path(TESTS_DATA_DIR_PATH / x) for x in expected[0]]
    assert test == [Path(TESTS_DATA_DIR_PATH / x) for x in expected[1]]


@pytest.mark.parametrize(
    ("dataset_paths", "test_size", "train_size", "glob_pattern", "split_in_each_directory", "shuffle", "expected"),
    [
        (
            ["dummy_datasets_01", "dummy_datasets_02"],
            None,
            None,
            "*.csv",
            False,
            False,
            (
                [
                    "dummy_datasets_01/dummy_data_000.csv",
                    "dummy_datasets_01/dummy_data_001.csv",
                    "dummy_datasets_01/dummy_data_003.csv",
                    "dummy_datasets_01/dummy_data_005.csv",
                    "dummy_datasets_01/dummy_data_006.csv",
                    "dummy_datasets_01/dummy_data_008.csv",
                    "dummy_datasets_01/dummy_data_009.csv",
                    "dummy_datasets_02/dummy_data_000.csv",
                    "dummy_datasets_02/dummy_data_002.csv",
                    "dummy_datasets_02/dummy_data_003.csv",
                    "dummy_datasets_02/dummy_data_004.csv",
                    "dummy_datasets_02/dummy_data_005.csv",
                    "dummy_datasets_02/dummy_data_006.csv",
                    "dummy_datasets_02/dummy_data_007.csv",
                    "dummy_datasets_02/dummy_data_009.csv",
                    "dummy_datasets_02/dummy_data_010.csv",
                    "dummy_datasets_02/dummy_data_011.csv",
                    "dummy_datasets_02/dummy_data_012.csv",
                    "dummy_datasets_02/dummy_data_013.csv",
                    "dummy_datasets_02/dummy_data_014.csv",
                    "dummy_datasets_02/dummy_data_015.csv",
                    "dummy_datasets_02/dummy_data_016.csv",
                    "dummy_datasets_02/dummy_data_018.csv",
                ],
                [
                    "dummy_datasets_01/dummy_data_002.csv",
                    "dummy_datasets_01/dummy_data_004.csv",
                    "dummy_datasets_01/dummy_data_007.csv",
                    "dummy_datasets_02/dummy_data_001.csv",
                    "dummy_datasets_02/dummy_data_008.csv",
                    "dummy_datasets_02/dummy_data_017.csv",
                    "dummy_datasets_02/dummy_data_019.csv",
                ],
            ),
        ),
        (
            ["dummy_datasets_02", "dummy_datasets_01"],
            None,
            None,
            "*.csv",
            True,
            False,
            (
                [
                    "dummy_datasets_02/dummy_data_000.csv",
                    "dummy_datasets_02/dummy_data_001.csv",
                    "dummy_datasets_02/dummy_data_003.csv",
                    "dummy_datasets_02/dummy_data_004.csv",
                    "dummy_datasets_02/dummy_data_005.csv",
                    "dummy_datasets_02/dummy_data_006.csv",
                    "dummy_datasets_02/dummy_data_008.csv",
                    "dummy_datasets_02/dummy_data_009.csv",
                    "dummy_datasets_02/dummy_data_010.csv",
                    "dummy_datasets_02/dummy_data_012.csv",
                    "dummy_datasets_02/dummy_data_013.csv",
                    "dummy_datasets_02/dummy_data_014.csv",
                    "dummy_datasets_02/dummy_data_016.csv",
                    "dummy_datasets_02/dummy_data_018.csv",
                    "dummy_datasets_02/dummy_data_019.csv",
                    "dummy_datasets_01/dummy_data_001.csv",
                    "dummy_datasets_01/dummy_data_003.csv",
                    "dummy_datasets_01/dummy_data_004.csv",
                    "dummy_datasets_01/dummy_data_005.csv",
                    "dummy_datasets_01/dummy_data_006.csv",
                    "dummy_datasets_01/dummy_data_007.csv",
                    "dummy_datasets_01/dummy_data_008.csv",
                    "dummy_datasets_01/dummy_data_009.csv",
                ],
                [
                    "dummy_datasets_02/dummy_data_002.csv",
                    "dummy_datasets_02/dummy_data_007.csv",
                    "dummy_datasets_02/dummy_data_011.csv",
                    "dummy_datasets_02/dummy_data_015.csv",
                    "dummy_datasets_02/dummy_data_017.csv",
                    "dummy_datasets_01/dummy_data_000.csv",
                    "dummy_datasets_01/dummy_data_002.csv",
                ],
            ),
        ),
    ],
)
def test_train_test_split_files_shuffle(
    dataset_paths: list[str],
    test_size: float | None,
    train_size: float | None,
    glob_pattern: str,
    *,
    split_in_each_directory: bool,
    shuffle: bool,
    expected: tuple[list[str], list[str]],
) -> None:
    seed = 123
    dirpaths = [TESTS_DATA_DIR_PATH / x for x in dataset_paths]
    train, test = train_test_split_files(
        dirpaths,
        test_size,
        train_size,
        glob_pattern,
        seed,
        shuffle=shuffle,
        split_in_each_directory=split_in_each_directory,
    )
    assert train == [Path(TESTS_DATA_DIR_PATH / x) for x in expected[0]]
    assert test == [Path(TESTS_DATA_DIR_PATH / x) for x in expected[1]]


# Local Variables:
# jinx-local-words: "csv ctrl dataset datasets dir nn noqa png sublabel symlink"
# End:
