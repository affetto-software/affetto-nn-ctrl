from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from affctrllib.logger import Logger

from affetto_nn_ctrl import TESTS_DIR_PATH
from affetto_nn_ctrl.control_utility import (
    MIN_UPDATE_Q_DELTA,
    WAIST_JOINT_INDEX,
    WAIST_JOINT_LIMIT,
    RandomTrajectory,
    get_rng,
    set_seed,
)
from affetto_nn_ctrl.event_logging import get_event_logger, start_event_logging

if TYPE_CHECKING:
    from collections.abc import Generator


DOF = 13
DEFAULT_UPDATE_T_RANGE = (0.5, 1.5)
DEFAULT_UPDATE_Q_RANGE = (20.0, 40.0)
DEFAULT_UPDATE_Q_LIMIT = (5.0, 95.0)
TESTS_DATA_DIR_PATH = TESTS_DIR_PATH / "data"


def assert_file_contents(expected: Path, actual: Path) -> None:
    from difflib import unified_diff

    with expected.open() as f:
        expected_lines = f.readlines()
    with actual.open() as f:
        actual_lines = f.readlines()
    diff = list(unified_diff(expected_lines, actual_lines))
    assert diff == [], "Unexpected file differences:\n" + "".join(diff)


@pytest.fixture(scope="class")
def make_work_directory() -> Generator[Path, Any, Any]:
    work_dir = TESTS_DIR_PATH / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    yield work_dir
    shutil.rmtree(work_dir)


def test_set_seed() -> None:
    set_seed(0)
    rng = get_rng()
    n = 10
    numbers1 = [rng.uniform(0.0, 1.0) for _ in range(n)]
    set_seed(0)
    rng = get_rng()
    numbers2 = [rng.uniform(0.0, 1.0) for _ in range(n)]
    assert numbers1 == numbers2


def test_get_rng() -> None:
    set_seed(0)
    rng1 = get_rng()
    rng2 = get_rng()
    assert rng1 is rng2


def test_get_rng_reset_seed() -> None:
    set_seed(0)
    rng1 = get_rng()
    set_seed(0)
    rng2 = get_rng()
    assert rng1 is not rng2


@pytest.fixture
def default_random_trajectory() -> RandomTrajectory:
    return RandomTrajectory(
        list(range(DOF)),
        np.zeros(DOF),
        0.0,
        DEFAULT_UPDATE_T_RANGE,
        DEFAULT_UPDATE_Q_RANGE,
        DEFAULT_UPDATE_Q_LIMIT,
    )


class TestRandomTrajectory:
    @pytest.mark.parametrize(
        ("active_joints", "update_t_range"),
        [
            ([0, 1], (0.5, 1.0)),
            ([1, 2, 3, 4], (0.1, 1.5)),
            (list(range(DOF)), (0.3, 1.1)),
        ],
    )
    def test_update_t_range_given_tuple(
        self,
        active_joints: list[int],
        update_t_range: tuple[float, float],
    ) -> None:
        rt = RandomTrajectory(
            active_joints,
            np.zeros(DOF),
            0.0,
            update_t_range,
            DEFAULT_UPDATE_Q_RANGE,
            DEFAULT_UPDATE_Q_LIMIT,
        )
        assert len(rt.update_t_range_list) == len(active_joints)
        for t_range in rt.update_t_range_list:
            assert t_range == update_t_range

    @pytest.mark.parametrize(
        ("active_joints", "update_t_range_list"),
        [
            ([4], [(0.1, 0.5)]),
            ([0, 1], [(0.5, 1.0), (0.5, 1.0)]),
            ([1, 2, 3, 4], [(0.1, 1.5), (0.2, 1.4), (0.3, 1.3), (0.4, 1.2)]),
        ],
    )
    def test_update_t_range_given_list(
        self,
        active_joints: list[int],
        update_t_range_list: list[tuple[float, float]],
    ) -> None:
        rt = RandomTrajectory(
            active_joints,
            np.zeros(DOF),
            0.0,
            update_t_range_list,
            DEFAULT_UPDATE_Q_RANGE,
            DEFAULT_UPDATE_Q_LIMIT,
            async_update=True,
        )
        assert len(rt.update_t_range_list) == len(active_joints)
        for i in range(len(active_joints)):
            assert rt.update_t_range_list[i] == update_t_range_list[i]

    def test_update_t_range_error_length_mismatch(self) -> None:
        active_joints = list(range(DOF))
        update_t_range_list = [(0.1, 0.5)]
        msg = r"Lengths of lists \(active joints / range list\) are mismatch: .*"
        with pytest.raises(ValueError, match=msg):
            _ = RandomTrajectory(
                active_joints,
                np.zeros(DOF),
                0.0,
                update_t_range_list,
                DEFAULT_UPDATE_Q_RANGE,
                DEFAULT_UPDATE_Q_LIMIT,
                async_update=False,
            )

    def test_update_t_range_warn_mismatch_when_sync(self) -> None:
        active_joints = [0, 1]
        update_t_range_list = [(0.1, 0.5), (0.2, 0.6)]
        with pytest.warns(UserWarning) as record:
            _ = RandomTrajectory(
                active_joints,
                np.zeros(DOF),
                0.0,
                update_t_range_list,
                DEFAULT_UPDATE_Q_RANGE,
                DEFAULT_UPDATE_Q_LIMIT,
                async_update=False,
            )
        assert len(record) == 1
        msg = "Enabled sync update but various update t range is given."
        assert str(record[0].message).startswith(msg)

    @pytest.mark.parametrize(
        ("active_joints", "update_q_range"),
        [
            ([0, 1], (20.0, 40.0)),
            ([1, 2, 3, 4], (10.0, 20.0)),
            (list(range(DOF)), (15.0, 30.0)),
        ],
    )
    def test_update_q_range_given_tuple(
        self,
        active_joints: list[int],
        update_q_range: tuple[float, float],
    ) -> None:
        rt = RandomTrajectory(
            active_joints,
            np.zeros(DOF),
            0.0,
            DEFAULT_UPDATE_T_RANGE,
            update_q_range,
            DEFAULT_UPDATE_Q_LIMIT,
        )
        assert len(rt.update_q_range_list) == len(active_joints)
        for q_range in rt.update_q_range_list:
            assert q_range == update_q_range

    @pytest.mark.parametrize(
        ("active_joints", "update_q_range_list"),
        [
            ([4], [(20.0, 40.0)]),
            ([0, 1], [(20.0, 40.0), (10.0, 200.0)]),
            ([1, 2, 3, 4], [(5.0, 15.0), (6.0, 14.0), (7.0, 13.0), (8.0, 12.0)]),
        ],
    )
    def test_update_q_range_given_list(
        self,
        active_joints: list[int],
        update_q_range_list: list[tuple[float, float]],
    ) -> None:
        rt = RandomTrajectory(
            active_joints,
            np.zeros(DOF),
            0.0,
            DEFAULT_UPDATE_T_RANGE,
            update_q_range_list,
            DEFAULT_UPDATE_Q_LIMIT,
        )
        assert len(rt.update_q_range_list) == len(active_joints)
        for i in range(len(active_joints)):
            assert rt.update_q_range_list[i] == update_q_range_list[i]

    def test_update_q_range_error_length_mismatch(self) -> None:
        active_joints = list(range(DOF))
        update_q_range_list = [(10.0, 90.0)]
        msg = r"Lengths of lists \(active joints / range list\) are mismatch: .*"
        with pytest.raises(ValueError, match=msg):
            _ = RandomTrajectory(
                active_joints,
                np.zeros(DOF),
                0.0,
                DEFAULT_UPDATE_T_RANGE,
                update_q_range_list,
                DEFAULT_UPDATE_Q_LIMIT,
            )

    @pytest.mark.parametrize(
        ("active_joints", "update_q_limit"),
        [
            ([1], (20.0, 80.0)),
            ([1, 2, 3, 4], (5.0, 95.0)),
            (list(range(1, DOF)), (10.0, 90.0)),
        ],
    )
    def test_update_q_limit_given_tuple(
        self,
        active_joints: list[int],
        update_q_limit: tuple[float, float],
    ) -> None:
        rt = RandomTrajectory(
            active_joints,
            np.zeros(DOF),
            0.0,
            DEFAULT_UPDATE_T_RANGE,
            DEFAULT_UPDATE_Q_RANGE,
            update_q_limit,
        )
        assert len(rt.update_q_limit_list) == len(active_joints)
        for q_range in rt.update_q_limit_list:
            assert q_range == update_q_limit

    @pytest.mark.parametrize(
        ("active_joints", "update_q_limit_list"),
        [
            ([0, 1], [(0.0, 100.0), (0.0, 100.0)]),
            ([1, 2, 3, 4], [(5.0, 95.0), (6.0, 94.0), (7.0, 93.0), (8.0, 92.0)]),
            ([0, 1, 2, 3, 4], [(4.0, 96.0), (5.0, 95.0), (6.0, 94.0), (7.0, 93.0), (8.0, 92.0)]),
            (list(range(DOF)), [(10.0, 90.0) for _ in range(DOF)]),
            (list(range(1, DOF)), [(10.0, 90.0) for _ in range(DOF - 1)]),
        ],
    )
    def test_update_q_limit_given_list(
        self,
        active_joints: list[int],
        update_q_limit_list: list[tuple[float, float]],
    ) -> None:
        rt = RandomTrajectory(
            active_joints,
            np.zeros(DOF),
            0.0,
            DEFAULT_UPDATE_T_RANGE,
            DEFAULT_UPDATE_Q_RANGE,
            update_q_limit_list,
        )
        assert len(rt.update_q_limit_list) == len(active_joints)
        for i in range(len(active_joints)):
            assert rt.update_q_limit_list[i] == update_q_limit_list[i]

    @pytest.mark.parametrize(
        ("active_joints", "update_q_limit_list"),
        [
            ([0, 1], (5.0, 95.0)),
            ([1, 2, 3], (5.0, 95.0)),
            (list(range(DOF)), (5.0, 95.0)),
            (list(range(1, DOF)), (5.0, 95.0)),
            ([1, 2, 0, 3], (5.0, 95.0)),
            ([1, 2, 3, 0], (5.0, 95.0)),
        ],
    )
    def test_update_q_limit_set_waist_joint_when_given_limit_as_tuple(
        self,
        active_joints: list[int],
        update_q_limit_list: tuple[float, float],
    ) -> None:
        rt = RandomTrajectory(
            active_joints,
            np.zeros(DOF),
            0.0,
            DEFAULT_UPDATE_T_RANGE,
            DEFAULT_UPDATE_Q_RANGE,
            update_q_limit_list,
        )
        assert len(rt.update_q_limit_list) == len(active_joints)
        for i, index in enumerate(active_joints):
            if index == WAIST_JOINT_INDEX:
                assert rt.update_q_limit_list[i] == WAIST_JOINT_LIMIT
            else:
                assert rt.update_q_limit_list[i] == update_q_limit_list

    def test_update_q_limit_error_length_mismatch(self) -> None:
        active_joints = list(range(DOF))
        update_q_limit_list = [(10.0, 90.0)]
        msg = r"Lengths of lists \(active joints / range list\) are mismatch: .*"
        with pytest.raises(ValueError, match=msg):
            _ = RandomTrajectory(
                active_joints,
                np.zeros(DOF),
                0.0,
                DEFAULT_UPDATE_T_RANGE,
                DEFAULT_UPDATE_Q_RANGE,
                update_q_limit_list,
            )

    @pytest.mark.parametrize(
        ("q_range", "q_limit"),
        [
            ((20.0, 40.0), (5.0, 95.0)),
            ((50.0, 60.0), (10.0, 90.0)),
            ((10.0, 15.0), (10.0, 90.0)),
        ],
    )
    def test_generate_new_position_ensure_range(
        self,
        default_random_trajectory: RandomTrajectory,
        q_range: tuple[float, float],
        q_limit: tuple[float, float],
    ) -> None:
        rt = default_random_trajectory
        n = 20
        rng = np.random.default_rng(int(datetime.now(tz=timezone.utc).timestamp()))
        for _ in range(n):
            q0 = rng.uniform(min(q_limit), max(q_limit))
            qdes = rt.generate_new_position(q0, q_range, q_limit, MIN_UPDATE_Q_DELTA)
            qdiff = abs(qdes - q0)
            assert MIN_UPDATE_Q_DELTA < qdiff < q_range[1]
            assert q_limit[0] < qdes < q_limit[1]


class TestRandomTrajectoryData:
    def make_output_path(self, base_directory: Path, active_joints: list[int], *, async_update: bool) -> Path:
        joints_str = "all" if len(active_joints) == DOF else "-".join(map(str, active_joints))
        sync_str = "async" if async_update else "sync"
        filename = f"rand_traj_{sync_str}_{joints_str}.dat"
        return base_directory / filename

    def generate_data(self, output: Path | None, active_joints: list[int], *, async_update: bool) -> None:
        set_seed(123456)
        total_duration = 20
        dt = 1e-3
        n_step = int(total_duration / dt)
        q0 = np.full(DOF, 50.0, dtype=float)
        rt = RandomTrajectory(
            active_joints,
            q0,
            0.0,
            DEFAULT_UPDATE_T_RANGE,
            DEFAULT_UPDATE_Q_RANGE,
            DEFAULT_UPDATE_Q_LIMIT,
            async_update=async_update,
        )
        logger = Logger()
        logger.set_labels("t", [f"q{i}" for i in range(DOF)], [f"dq{i}" for i in range(DOF)])
        qdes, dqdes = rt.get_qdes_func(), rt.get_dqdes_func()
        t: float = 0.0
        for i in range(n_step + 1):
            t = i * total_duration / n_step
            logger.store(t, qdes(t), dqdes(t))
        logger.dump(output, quiet=True)

    @pytest.mark.parametrize(
        "active_joints",
        [
            [0, 1, 2],
            # [1, 3, 5, 7],
            # list(range(DOF)),
        ],
    )
    # @pytest.mark.parametrize("async_update", [False, True])
    @pytest.mark.parametrize("async_update", [False])
    def test_check_trajectory_sync(
        self,
        make_work_directory: Path,
        active_joints: list[int],
        async_update: bool,  # noqa: FBT001
    ) -> None:
        output = self.make_output_path(make_work_directory, active_joints, async_update=async_update)
        self.generate_data(output, active_joints, async_update=async_update)
        assert_file_contents(TESTS_DATA_DIR_PATH / output.name, output)


def generate_data(active_joints: list[int], *, async_update: bool) -> Path:
    event_logger = get_event_logger()
    generator = TestRandomTrajectoryData()
    output = generator.make_output_path(TESTS_DATA_DIR_PATH, active_joints, async_update=async_update)
    output.parent.mkdir(parents=True, exist_ok=True)
    generator.generate_data(output, active_joints, async_update=async_update)
    if event_logger:
        event_logger.info("Data generated: %s", output)
    return output


def plot_generated_data(output: Path) -> None:
    import matplotlib.pyplot as plt
    from pyplotutil.datautil import Data

    event_logger = get_event_logger()

    # Setup
    xlim: tuple[float, float] | None = None
    xlim = (0, 5)
    show_legend = False
    data = Data(output)
    if event_logger:
        event_logger.info("Data loaded: %s", output)

    # Make plot of joint position
    plt.figure(1)
    for i in range(DOF):
        plt.plot(data.t, getattr(data, f"q{i}"), label=f"{i}")
    plt.axhline(DEFAULT_UPDATE_Q_LIMIT[0], ls="--", color="k")
    plt.axhline(DEFAULT_UPDATE_Q_LIMIT[1], ls="--", color="k")
    plt.xlabel("time [s]")
    plt.ylabel("joint position [0-100]")
    plt.title("Joint position")
    plt.ylim((0, 100))
    plt.xlim(xlim)
    if show_legend:
        plt.legend()

    # Make plot of joint velocity
    plt.figure(2)
    for i in range(DOF):
        plt.plot(data.t, getattr(data, f"dq{i}"), label=f"{i}")
    plt.xlabel("time [s]")
    plt.ylabel("joint velocity [0-100/s]")
    plt.title("Joint velocity")
    plt.xlim(xlim)
    if show_legend:
        plt.legend()

    plt.show()


def main() -> None:
    import sys

    start_event_logging(sys.argv, logging_level="INFO")
    if len(sys.argv) == 1:
        active_joints = [0, 1, 2]
        async_update = False
        output = generate_data(active_joints, async_update=async_update)
    else:
        output = Path(sys.argv[1])
    plot_generated_data(output)


if __name__ == "__main__":
    main()


# Local Variables:
# jinx-local-words: "async dat dq noqa traj"
# End:
