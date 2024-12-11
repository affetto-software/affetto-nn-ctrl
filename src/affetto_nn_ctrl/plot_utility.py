from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from pyplotutil.datautil import Data

from affetto_nn_ctrl._typing import NoDefault, no_default
from affetto_nn_ctrl.data_handling import find_latest_data_dir_path
from affetto_nn_ctrl.event_logging import event_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


DEFAULT_JOINT_NAMES = {
    "0": "waist",
    "1": "left_shoulder_elevation",
    "2": "left_shoulder_abduction",
    "3": "left_shoulder_flexion",
    "4": "left_elbow_rotation",
    "5": "left_elbow_flexion",
    "6": "left_wrist_supination",
    "7": "right_shoulder_elevation",
    "8": "right_shoulder_abduction",
    "9": "right_shoulder_flexion",
    "10": "right_elbow_rotation",
    "11": "right_elbow_flexion",
    "12": "right_wrist_supination",
}


def savefig(
    fig: Figure,
    data_file_path: Path,
    ext_list: list[str],
    *,
    dpi: float | str = "figure",
    separate_dir: bool = True,
) -> list[Path]:
    saved_figpath: list[Path] = []
    for ext in ext_list:
        e = ext
        if not e.startswith("."):
            e = f".{e}"
        if data_file_path.suffix.startswith(".") and data_file_path.suffix[1].isdigit():
            # When data_file_path.suffix starts with a digit it's not a suffix.
            # >>> Path('awesome_ratio-2.5').with_suffix('.svg')
            # 'awesome_ratio-2.svg'  # this is wrong filename
            data_file_path = data_file_path.with_name(data_file_path.name + ".x")
        figpath = data_file_path.with_suffix(e)
        if separate_dir:
            # Separate saving directories across extensions.
            figpath = figpath.parent / e[1:] / figpath.name
            figpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figpath, dpi=dpi, bbox_inches="tight")
        saved_figpath.append(figpath)
        msg = f"Figure saved: {figpath}"
        event_logger().info(msg)
    return saved_figpath


FILENAME_T = TypeVar("FILENAME_T", str, Path)


def _compatible_filename(filename: FILENAME_T) -> FILENAME_T:
    table = {":": "", " ": "_", "(": "", ")": "", "+": "x"}
    compat_filename = str(filename).translate(str.maketrans(table))  # type: ignore[arg-type]
    event_logger().debug("Making filename compatible to file system:")
    event_logger().debug("       given filename: %s", filename)
    event_logger().debug("  compatible filename: %s", compat_filename)
    return type(filename)(compat_filename)


def save_figure(
    fig: Figure,
    save_dir_path: Path,
    basename: str,
    ext_list: list[str] | None,
    *,
    loaded_from: str | None | NoDefault = no_default,
    dpi: float | str = "figure",
) -> list[Path]:
    if save_dir_path is None:
        msg = f"'None' is not allowed for directory path: {save_dir_path}, {type(save_dir_path)}"
        raise ValueError(msg)

    if loaded_from is no_default:
        try:
            import __main__

            loaded_from = Path(__main__.__file__).stem
        except ImportError:
            loaded_from = None

    built_filename = save_dir_path
    if loaded_from:
        built_filename /= loaded_from
    built_filename /= basename
    built_filename = _compatible_filename(built_filename)

    fig.tight_layout()
    saved_figures: list[Path] = []
    if ext_list is not None:
        saved_figures = savefig(fig, built_filename, ext_list, dpi=dpi)
    else:
        event_logger().warning("Nothing saved.")
        event_logger().debug("Figures have not been saved since no extension is provided.")
    return saved_figures


def load_dataset(dataset_dir_path: Path, pattern: str = "**.csv") -> list[Data]:
    if not dataset_dir_path.exists():
        msg = f"Directory does not exist: {dataset_dir_path}"
        raise ValueError(msg)
    csv_files = dataset_dir_path.glob(pattern)
    dataset = [Data(csv) for csv in csv_files]
    if len(dataset) == 0:
        msg = f"No CSV files found: {dataset_dir_path}"
        raise RuntimeError(msg)
    return dataset


def load_latest_dataset(dataset_dir_path: Path, pattern: str = "**.csv") -> list[Data]:
    return load_dataset(find_latest_data_dir_path(dataset_dir_path, force_find_by_pattern=True), pattern)


def pickup_datapath(path_list: list[Path], pickup_list: list[int | str]) -> list[Path]:
    cherries = []
    for i in pickup_list:
        if isinstance(i, str):
            if "-" in i:
                begin, end = map(int, i.split("-"))
                cherries.extend(path_list[begin : end + 1])
            else:
                cherries.append(path_list[int(i)])
        else:
            cherries.append(path_list[i])
    return cherries


def find_minimum_length(dataset: list[Data]) -> int:
    # find minimum length among data files in dataset
    n = 2**63 - 1
    for data in dataset:
        n = min(n, len(data))
    return n


def extract_all_values(dataset: list[Data], key: str) -> np.ndarray:
    n = find_minimum_length(dataset)
    data_array = np.vstack([getattr(data, key)[:n] for data in dataset])
    assert data_array.shape == (len(dataset), n)
    return data_array


def extract_common_parts(*paths: str | Path) -> Path:
    # Convert paths to Path objects
    path_objects = [Path(path) for path in paths]

    # Find the shortest path
    shortest_path = min(path_objects, key=lambda p: len(p.parts))

    # Iterate over the shortest path's parts
    common_parts = []
    for i, part in enumerate(shortest_path.parts):
        if all(part in path.parts[: i + 1] for path in path_objects):
            if part == "/":
                common_parts.append("")
            else:
                common_parts.append(part)
        else:
            break

    # Join common parts back into a path string
    common_path = "/".join(common_parts)

    return Path(common_path)


def get_tlim_mask(t: np.ndarray, tlim: tuple[float, float] | None) -> np.ndarray:
    return np.full(t.shape, fill_value=True) if tlim is None else (t >= tlim[0]) & (t <= tlim[1])


def mask_data(tlim: tuple[float, float] | None, t: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert len(t) == len(data)
    mask = get_tlim_mask(t, tlim)
    return t[mask], data[mask]


def plot_multi_timeseries(
    ax: Axes,
    t: np.ndarray,
    data_array: np.ndarray,
    lw: int,
    c: str | None = None,
    labels: Iterable[str] | None = None,
    cmap_name: str = "tab10",
) -> None:
    data_array = np.atleast_2d(data_array)
    cmap = plt.get_cmap(cmap_name)
    if labels is None:
        labels = ["" for _ in range(len(data_array))]
    for i, (data, label) in enumerate(zip(data_array, labels, strict=True)):
        color = cmap(i) if c is None else c
        n = min(len(t), len(data))
        ax.plot(t[:n], data[:n], label=str(label), lw=lw, c=color)


def calculate_mean_err(
    data_array: np.ndarray,
    err_type: str = "std",
    ddof: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    mean = np.mean(data_array, axis=0)
    if err_type.lower() in ("std", "sd"):
        # standard deviation
        std = np.std(data_array, axis=0, ddof=ddof)
        return mean, std, None

    if err_type.lower() in ("var",):
        # variance
        var = np.var(data_array, axis=0, ddof=ddof)
        return mean, var, None

    if err_type.lower() in ("range",):
        # range
        lower = mean - np.min(data_array, axis=0)
        upper = np.max(data_array, axis=0) - mean
        return mean, lower, upper

    if err_type.lower() in ("se",):
        # standard error
        std = np.std(data_array, axis=0, ddof=ddof)
        se = std / np.sqrt(len(std))
        return mean, se, None

    if err_type.lower() in ("ci",):
        # confidence interval
        raise NotImplementedError

    msg = f"unrecognized error type: {err_type}"
    raise ValueError(msg)


# Local Variables:
# jinx-local-words: "arg ci csv dataset sd se"
# End:
