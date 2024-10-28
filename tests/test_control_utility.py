from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from affctrllib.logger import Logger
from numpy.testing import assert_array_equal, assert_raises

from affetto_nn_ctrl import TESTS_DIR_PATH
from affetto_nn_ctrl.control_utility import (
    MIN_UPDATE_Q_DELTA,
    WAIST_JOINT_INDEX,
    WAIST_JOINT_LIMIT,
    RandomTrajectory,
    RobotInitializer,
    get_rng,
    set_seed,
)
from affetto_nn_ctrl.event_logging import get_event_logger, start_event_logging

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


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
def default_robot_initializer() -> RobotInitializer:
    return RobotInitializer(13, duration=5.0, manner="position", q_init=50.0)


class TestRobotInitializer:
    def test_init_defaults(self) -> None:
        dof = 13
        init = RobotInitializer(dof)
        assert init.dof == dof
        assert init.duration == RobotInitializer.DEFAULT_DURATION
        assert init.get_manner() == "position"
        assert_array_equal(init.get_q_init(), np.full((dof,), RobotInitializer.DEFAULT_Q_INIT))
        assert_array_equal(init.get_ca_init(), np.full((dof,), RobotInitializer.DEFAULT_CA_INIT))
        assert_array_equal(init.get_cb_init(), np.full((dof,), RobotInitializer.DEFAULT_CB_INIT))

    @pytest.mark.parametrize("dof", [1, 4, 13])
    def test_init_dof(self, dof: int) -> None:
        init = RobotInitializer(dof)
        assert init.dof == dof

    @pytest.mark.parametrize("duration", [3, 5, 10])
    def test_set_duration(self, duration: float) -> None:
        init = RobotInitializer(13)
        init.duration = duration
        assert init.duration == duration

    @pytest.mark.parametrize(
        ("manner", "expected"),
        [
            ("position", "position"),
            ("pos", "position"),
            ("p", "position"),
            ("q", "position"),
            ("pressure", "pressure"),
            ("pre", "pressure"),
            ("pres", "pressure"),
            ("valve", "pressure"),
            ("v", "pressure"),
        ],
    )
    def test_set_manner(self, manner: str, expected: str) -> None:
        init = RobotInitializer(13)
        init.set_manner(manner)
        assert init.get_manner() == expected

    def test_set_manner_error_invalid_value(self) -> None:
        init = RobotInitializer(13)
        invalid_manner = "invalid_manner"
        msg = f"Unrecognized manner for RobotInitializer: {invalid_manner}"
        with pytest.raises(ValueError, match=msg):
            init.set_manner(invalid_manner)

    @pytest.mark.parametrize(("dof", "q_init"), [(1, 10.0), (2, 15.0), (10, 50), (13, 75)])
    def test_set_q_init_float(self, dof: int, q_init: float) -> None:
        init = RobotInitializer(dof)
        init.set_q_init(q_init)
        expected = np.full((dof,), q_init)
        assert_array_equal(init.get_q_init(), expected)
        assert init.get_q_init().dtype == float

    @pytest.mark.parametrize(
        ("dof", "q_init", "expected"),
        [
            (1, [10], np.array([10.0])),
            (2, (10, 20), np.array([10.0, 20.0])),
            (3, np.array([60, 60, 60]), np.array([60, 60, 60])),
            (2, (30,), np.array([30.0, 30.0])),
            (3, (75.0,), np.array([75.0, 75.0, 75.0])),
        ],
    )
    def test_set_q_init_array(self, dof: int, q_init: Sequence[float], expected: np.ndarray) -> None:
        init = RobotInitializer(dof)
        init.set_q_init(q_init)
        assert_array_equal(init.get_q_init(), expected)
        assert init.get_q_init().dtype == float

    @pytest.mark.parametrize(
        ("dof", "q_init"),
        [
            (1, (10, 20)),
            (3, (10, 20)),
            (3, (10, 20, 30, 40)),
        ],
    )
    def test_set_q_init_error_size_mismatch(self, dof: int, q_init: Sequence[float]) -> None:
        init = RobotInitializer(dof)
        msg = rf"Unable to set values due to size mismatch: dof={dof}, given_value={q_init}"
        with pytest.raises(ValueError, match=re.escape(msg)):
            init.set_q_init(q_init)

    @pytest.mark.parametrize(("dof", "ca_init"), [(1, 0.0), (2, 15.0), (10, 100), (13, 125)])
    def test_set_ca_init_float(self, dof: int, ca_init: float) -> None:
        init = RobotInitializer(dof)
        init.set_ca_init(ca_init)
        expected = np.full((dof,), ca_init)
        assert_array_equal(init.get_ca_init(), expected)
        assert init.get_ca_init().dtype == float

    @pytest.mark.parametrize(
        ("dof", "ca_init", "expected"),
        [
            (1, [0], np.array([0.0])),
            (2, (10, 20), np.array([10.0, 20.0])),
            (3, np.array([100, 90, 80]), np.array([100.0, 90.0, 80.0])),
            (2, (250,), np.array([250.0, 250.0])),
            (3, (175.0,), np.array([175.0, 175.0, 175.0])),
        ],
    )
    def test_set_ca_init_array(self, dof: int, ca_init: Sequence[float], expected: np.ndarray) -> None:
        init = RobotInitializer(dof)
        init.set_ca_init(ca_init)
        assert_array_equal(init.get_ca_init(), expected)
        assert init.get_ca_init().dtype == float

    @pytest.mark.parametrize(
        ("dof", "ca_init"),
        [
            (1, (0, 200)),
            (3, (0, 200)),
            (3, (0, 50, 100, 200)),
        ],
    )
    def test_set_ca_init_error_size_mismatch(self, dof: int, ca_init: Sequence[float]) -> None:
        init = RobotInitializer(dof)
        msg = rf"Unable to set values due to size mismatch: dof={dof}, given_value={ca_init}"
        with pytest.raises(ValueError, match=re.escape(msg)):
            init.set_ca_init(ca_init)

    @pytest.mark.parametrize(("dof", "cb_init"), [(1, 10.0), (2, 10.0), (10, 90), (13, 130)])
    def test_set_cb_init_float(self, dof: int, cb_init: float) -> None:
        init = RobotInitializer(dof)
        init.set_cb_init(cb_init)
        expected = np.full((dof,), cb_init)
        assert_array_equal(init.get_cb_init(), expected)
        assert init.get_cb_init().dtype == float

    @pytest.mark.parametrize(
        ("dof", "cb_init", "expected"),
        [
            (1, [0], np.array([0.0])),
            (2, (20, 30), np.array([20.0, 30.0])),
            (3, np.array([120, 130, 140]), np.array([120.0, 130.0, 140.0])),
            (2, (255,), np.array([255.0, 255.0])),
            (3, (150.0,), np.array([150.0, 150.0, 150.0])),
        ],
    )
    def test_set_cb_init_array(self, dof: int, cb_init: Sequence[float], expected: np.ndarray) -> None:
        init = RobotInitializer(dof)
        init.set_cb_init(cb_init)
        assert_array_equal(init.get_cb_init(), expected)
        assert init.get_cb_init().dtype == float

    @pytest.mark.parametrize(
        ("dof", "cb_init"),
        [
            (1, (0, 200)),
            (3, (0, 200)),
            (3, (0, 50, 100, 200)),
        ],
    )
    def test_set_cb_init_error_size_mismatch(self, dof: int, cb_init: Sequence[float]) -> None:
        init = RobotInitializer(dof)
        msg = rf"Unable to set values due to size mismatch: dof={dof}, given_value={cb_init}"
        with pytest.raises(ValueError, match=re.escape(msg)):
            init.set_cb_init(cb_init)

    def test_load_config(self) -> None:
        dof = 13
        config = TESTS_DATA_DIR_PATH / "affetto.toml"
        init = RobotInitializer(dof)
        init.load_config(config)
        expected_duration = 7
        expected_manner = "pressure"
        expected_q_init = np.full((dof,), 45.0, dtype=float)
        expected_ca_init = np.full((dof,), 10.0, dtype=float)
        expected_cb_init = np.full((dof,), 100.0, dtype=float)
        assert init.duration == expected_duration
        assert init.get_manner() == expected_manner
        assert_array_equal(init.get_q_init(), expected_q_init)
        assert_array_equal(init.get_ca_init(), expected_ca_init)
        assert_array_equal(init.get_cb_init(), expected_cb_init)

    def test_load_config_set_default_when_no_entry(self) -> None:
        dof = 13
        config = TESTS_DATA_DIR_PATH / "dummy.toml"
        init = RobotInitializer(dof)
        init.load_config(config)
        expected_q_init = np.full((dof,), RobotInitializer.DEFAULT_Q_INIT, dtype=float)
        expected_ca_init = np.full((dof,), RobotInitializer.DEFAULT_CA_INIT, dtype=float)
        expected_cb_init = np.full((dof,), RobotInitializer.DEFAULT_CB_INIT, dtype=float)
        assert init.duration == RobotInitializer.DEFAULT_DURATION
        assert init.get_manner() == RobotInitializer.DEFAULT_MANNER
        assert_array_equal(init.get_q_init(), expected_q_init)
        assert_array_equal(init.get_ca_init(), expected_ca_init)
        assert_array_equal(init.get_cb_init(), expected_cb_init)

    def test_init_update_by_config(self) -> None:
        dof = 13
        config = TESTS_DATA_DIR_PATH / "affetto.toml"
        init = RobotInitializer(dof, config=config)

        # Confirm that the following expected values match values defined in "tests/data/affetto.toml"
        ########
        expected_duration = 7
        expected_manner = "pressure"
        expected_q_init = np.full((dof,), 45.0, dtype=float)
        expected_ca_init = np.full((dof,), 10.0, dtype=float)
        expected_cb_init = np.full((dof,), 100.0, dtype=float)
        ########

        assert init.duration == expected_duration
        assert init.get_manner() == expected_manner
        assert_array_equal(init.get_q_init(), expected_q_init)
        assert_array_equal(init.get_ca_init(), expected_ca_init)
        assert_array_equal(init.get_cb_init(), expected_cb_init)

    def test_init_update_by_arguments(self) -> None:
        dof = 13
        config = TESTS_DATA_DIR_PATH / "affetto.toml"
        expected_duration = 12
        expected_manner = "pressure"
        expected_q_init = np.full((dof,), 55.0, dtype=float)
        expected_cb_init = np.full((dof,), 155.0, dtype=float)
        init = RobotInitializer(
            dof,
            config=config,
            duration=expected_duration,
            manner=expected_manner,
            q_init=expected_q_init,
            # ca_init=expected_ca_init, # ca_init intentionally omitted # noqa: ERA001
            cb_init=expected_cb_init,
        )
        assert init.duration == expected_duration
        assert init.get_manner() == expected_manner
        assert_array_equal(init.get_q_init(), expected_q_init)
        assert_array_equal(init.get_cb_init(), expected_cb_init)

        # Confirm that the following expected values match values defined in "tests/data/affetto.toml"
        ########
        expected_ca_init = np.full((dof,), 10.0, dtype=float)
        ########
        assert_array_equal(init.get_ca_init(), expected_ca_init)


@pytest.fixture
def default_random_trajectory() -> RandomTrajectory:
    return RandomTrajectory(
        list(range(DOF)),
        0.0,
        np.zeros(DOF),
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
            0.0,
            np.zeros(DOF),
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
            0.0,
            np.zeros(DOF),
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
                0.0,
                np.zeros(DOF),
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
                0.0,
                np.zeros(DOF),
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
            0.0,
            np.zeros(DOF),
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
            0.0,
            np.zeros(DOF),
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
                0.0,
                np.zeros(DOF),
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
            0.0,
            np.zeros(DOF),
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
            0.0,
            np.zeros(DOF),
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
            0.0,
            np.zeros(DOF),
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
                0.0,
                np.zeros(DOF),
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


@pytest.fixture(scope="class")
def make_work_directory() -> Generator[Path, Any, Any]:
    work_dir = TESTS_DIR_PATH / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    yield work_dir
    shutil.rmtree(work_dir)


class TestRandomTrajectoryData:
    def make_output_path(
        self,
        base_directory: Path,
        active_joints: list[int],
        update_profile: str,
        *,
        async_update: bool,
    ) -> Path:
        joints_str = "all" if len(active_joints) == DOF else "-".join(map(str, active_joints))
        sync_str = "async" if async_update else "sync"
        filename = f"rand_traj_{update_profile}_{sync_str}_{joints_str}.csv"
        return base_directory / filename

    def generate_data(
        self,
        output: Path | None,
        active_joints: list[int],
        update_profile: str,
        *,
        async_update: bool,
    ) -> None:
        total_duration = 10
        dt = 1e-2
        n_step = int(total_duration / dt)
        q0 = np.full(DOF, 50.0, dtype=float)
        rt = RandomTrajectory(
            active_joints,
            0.0,
            q0,
            DEFAULT_UPDATE_T_RANGE,
            DEFAULT_UPDATE_Q_RANGE,
            DEFAULT_UPDATE_Q_LIMIT,
            update_profile=update_profile,
            async_update=async_update,
            seed=123456,
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
            [1, 3, 5, 7],
            list(range(DOF)),
        ],
    )
    @pytest.mark.parametrize(
        "update_profile",
        [
            "trapez",
            "step",
        ],
    )
    @pytest.mark.parametrize("async_update", [False, True])
    def test_check_trajectory_sync(
        self,
        make_work_directory: Path,
        active_joints: list[int],
        update_profile: str,
        async_update: bool,  # noqa: FBT001
    ) -> None:
        output = self.make_output_path(make_work_directory, active_joints, update_profile, async_update=async_update)
        self.generate_data(output, active_joints, update_profile, async_update=async_update)
        assert_file_contents(TESTS_DATA_DIR_PATH / output.name, output)

    def test_reset_updater(self) -> None:
        active_joints = [0, 1, 2]
        q0 = np.full((DOF,), 50.0, dtype=float)
        rt = RandomTrajectory(
            active_joints,
            0.0,
            q0,
            DEFAULT_UPDATE_T_RANGE,
            DEFAULT_UPDATE_Q_RANGE,
            DEFAULT_UPDATE_Q_LIMIT,
        )
        n = 10
        dt = 1e-3
        qdes, dqdes = rt.get_qdes_func(), rt.get_dqdes_func()

        # Generate random trajectory
        data_q_1: list[np.ndarray] = []
        data_dq_1: list[np.ndarray] = []
        for i in range(n):
            t = i * dt
            data_q_1.append(qdes(t))
            data_dq_1.append(dqdes(t))
        # Reset generator
        rt.reset_updater()
        # Generate random trajectory again
        data_q_2: list[np.ndarray] = []
        data_dq_2: list[np.ndarray] = []
        for i in range(n):
            t = i * dt
            data_q_2.append(qdes(t))
            data_dq_2.append(dqdes(t))

        # First values are equal
        assert_array_equal(data_q_1[0], data_q_2[0])
        assert_array_equal(data_dq_1[0], data_dq_2[0])
        # Then values are diversified
        for i in range(1, n):
            assert_raises(AssertionError, assert_array_equal, data_q_1[i], data_q_2[i])
            assert_raises(AssertionError, assert_array_equal, data_dq_1[i], data_dq_2[i])


def generate_data(active_joints: list[int], update_profile: str, *, async_update: bool) -> Path:
    event_logger = get_event_logger()
    generator = TestRandomTrajectoryData()
    output = generator.make_output_path(TESTS_DATA_DIR_PATH, active_joints, update_profile, async_update=async_update)
    output.parent.mkdir(parents=True, exist_ok=True)
    generator.generate_data(output, active_joints, update_profile, async_update=async_update)
    if event_logger:
        event_logger.info("Data generated: %s", output)
    return output


def plot_generated_data(output: Path, xlim: tuple[float, float] | None = None, *, show_legend: bool = True) -> None:
    import matplotlib.pyplot as plt
    from pyplotutil.datautil import Data

    event_logger = get_event_logger()

    # Setup
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


def check_generated_data(active_joints: list[int], update_profile: str, *, async_update: bool, show_plot: bool) -> None:
    output = generate_data(active_joints, update_profile, async_update=async_update)
    if show_plot:
        plot_generated_data(output, xlim=None, show_legend=False)


def generate_expected_data(*, show_plot: bool = True) -> None:
    active_joints_list: list[list[int]] = [
        [0, 1, 2],
        [1, 3, 5, 7],
        list(range(DOF)),
    ]
    update_profile_list = [
        "trapez",
        "step",
    ]
    async_update_list: list[bool] = [
        False,
        True,
    ]
    for active_joints in active_joints_list:
        for update_profile in update_profile_list:
            for async_update in async_update_list:
                check_generated_data(active_joints, update_profile, async_update=async_update, show_plot=show_plot)


def main() -> None:
    import sys

    start_event_logging(sys.argv, logging_level="INFO")
    if len(sys.argv) == 1:
        generate_expected_data(show_plot=True)
    else:
        output = Path(sys.argv[1])
        plot_generated_data(output)


if __name__ == "__main__":
    main()


# Local Variables:
# jinx-local-words: "async cb const csv dat dof dq init noqa pos traj trapez"
# End:
