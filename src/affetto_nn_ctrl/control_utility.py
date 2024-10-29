# ruff: noqa: S311
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from affctrllib import PTP, AffComm, AffPosCtrl, AffStateThread, Logger, Timer
from numpy.random import Generator, default_rng

from affetto_nn_ctrl.event_logging import get_event_logger

if sys.version_info < (3, 11):
    import tomli as tomllib  # type: ignore[reportMissingImport,import-not-found]
else:
    import tomllib

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from affetto_nn_ctrl import CONTROLLER_T


__SEED: int | None = None
__GLOBAL_RNG: Generator = default_rng()


class GetGlobalRngT:
    pass


GET_GLOBAL_RNG = GetGlobalRngT()


def set_seed(seed: int | None) -> None:
    global __SEED  # noqa: PLW0603
    global __GLOBAL_RNG  # noqa: PLW0603
    if isinstance(seed, int | None):
        __SEED = seed
        __GLOBAL_RNG = default_rng(__SEED)
        np.random.seed(__SEED)  # noqa: NPY002
    else:
        msg = f"`seed` is expected to be an `int` or `None`, not {type(seed)}"
        raise TypeError(msg)


def get_seed() -> int | None:
    return __SEED


def get_rng(seed: int | Generator | GetGlobalRngT | None = GET_GLOBAL_RNG) -> Generator:
    if isinstance(seed, GetGlobalRngT):
        return __GLOBAL_RNG
    if isinstance(seed, Generator):
        return seed
    if isinstance(seed, int | None):
        return default_rng(seed)
    msg = f"`seed` is expected to be an `int` or `None`, not {type(seed)}"
    raise TypeError(msg)


MIN_UPDATE_Q_DELTA = 1e-4
WAIST_JOINT_INDEX = 0
WAIST_JOINT_LIMIT = (40.0, 60.0)


def create_controller(
    config: str,
    sfreq: float | None,
    cfreq: float | None,
    waiting_time: float = 5.0,
) -> CONTROLLER_T:
    event_logger = get_event_logger()
    if event_logger:
        event_logger.info("Loaded config: %s", config)
        event_logger.debug("sensor frequency: %s, control frequency: %s", sfreq, cfreq)

    comm = AffComm(config_path=config)
    comm.create_command_socket()
    state = AffStateThread(config=config, freq=sfreq, logging=False, output=None, butterworth=True)
    ctrl = AffPosCtrl(config_path=config, freq=cfreq)
    state.prepare()
    state.start()
    if event_logger:
        event_logger.debug("Controller created.")

    if event_logger:
        event_logger.info("Waiting until robot gets stationary for %s s...", waiting_time)
    time.sleep(waiting_time)

    return comm, ctrl, state


def create_default_logger(dof: int) -> Logger:
    event_logger = get_event_logger()
    logger = Logger()
    logger.set_labels(
        "t",
        # raw data
        [f"rq{i}" for i in range(dof)],
        [f"rdq{i}" for i in range(dof)],
        [f"rpa{i}" for i in range(dof)],
        [f"rpb{i}" for i in range(dof)],
        # estimated states
        [f"q{i}" for i in range(dof)],
        [f"dq{i}" for i in range(dof)],
        [f"pa{i}" for i in range(dof)],
        [f"pb{i}" for i in range(dof)],
        # command data
        [f"ca{i}" for i in range(dof)],
        [f"cb{i}" for i in range(dof)],
        [f"qdes{i}" for i in range(dof)],
        [f"dqdes{i}" for i in range(dof)],
    )
    if event_logger:
        event_logger.debug("Default logger created.")

    return logger


def create_const_trajectory(
    qdes: float | list[float] | np.ndarray,
    joint: int | list[int],
    q0: np.ndarray,
) -> tuple[Callable[[float], np.ndarray], Callable[[float], np.ndarray]]:
    def qdes_func(_: float) -> np.ndarray:
        q = np.copy(q0)
        q[0] = 50  # make waist joint keep at middle.
        q[joint] = qdes
        return q

    def dqdes_func(_: float) -> np.ndarray:
        return np.zeros(len(q0), dtype=float)

    return qdes_func, dqdes_func


def reset_logger(logger: Logger | None, log_filename: str | Path | None) -> Logger | None:
    event_logger = get_event_logger()
    if logger is not None:
        logger.erase_data()
        if event_logger:
            event_logger.debug("Logger data has been erased.")
        if log_filename is not None:
            logger.set_filename(log_filename)
            if event_logger:
                event_logger.debug("Logger filename is updated: %s", log_filename)
    return logger


def _select_time_updater(timer: Timer, time_updater: str) -> Callable[[], float]:
    current_time_func: Callable[[], float]
    if time_updater == "elapsed":
        current_time_func = timer.elapsed_time
    elif time_updater == "accumulated":
        current_time_func = timer.accumulated_time
    else:
        msg = f"unrecognized time updater: {time_updater}"
        raise ValueError(msg)
    return current_time_func


def control_position(
    controller: CONTROLLER_T,
    qdes_func: Callable[[float], np.ndarray | float],
    dqdes_func: Callable[[float], np.ndarray | float],
    duration: float,
    logger: Logger | None = None,
    log_filename: str | Path | None = None,
    time_updater: str = "elapsed",
    header_text: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    reset_logger(logger, log_filename)
    comm, ctrl, state = controller
    ca, cb = np.zeros(ctrl.dof, dtype=float), np.zeros(ctrl.dof, dtype=float)
    timer = Timer(rate=ctrl.freq)
    current_time = _select_time_updater(timer, time_updater)

    timer.start()
    t = 0.0
    while t < duration:
        sys.stdout.write(f"\r{header_text} [{t:6.2f}/{duration:.2f}]")
        t = current_time()
        rq, rdq, rpa, rpb = state.get_raw_states()
        q, dq, pa, pb = state.get_states()
        qdes = qdes_func(t)
        dqdes = dqdes_func(t)
        ca, cb = ctrl.update(t, q, dq, pa, pb, qdes, dqdes)
        comm.send_commands(ca, cb)
        if logger is not None:
            logger.store(t, rq, rdq, rpa, rpb, q, dq, pa, pb, ca, cb, qdes, dqdes)
        timer.block()
    sys.stdout.write("\n")
    # Return the last commands that have been sent to the valve.
    return ca, cb


def control_pressure(
    controller: CONTROLLER_T,
    ca_func: Callable[[float], np.ndarray | float],
    cb_func: Callable[[float], np.ndarray | float],
    duration: float,
    logger: Logger | None = None,
    log_filename: str | Path | None = None,
    time_updater: str = "elapsed",
    header_text: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    reset_logger(logger, log_filename)
    comm, ctrl, state = controller
    ca, cb = np.zeros(ctrl.dof, dtype=float), np.zeros(ctrl.dof, dtype=float)
    timer = Timer(rate=ctrl.freq)
    dummy = np.asarray([-1.0 for _ in range(ctrl.dof)])
    current_time = _select_time_updater(timer, time_updater)

    timer.start()
    t = 0.0
    while t < duration:
        sys.stdout.write(f"\r{header_text} [{t:6.2f}/{duration:.2f}]")
        t = current_time()
        rq, rdq, rpa, rpb = state.get_raw_states()
        q, dq, pa, pb = state.get_states()
        ca = np.asarray(ca_func(t))
        cb = np.asarray(cb_func(t))
        comm.send_commands(ca, cb)
        if logger is not None:
            logger.store(t, rq, rdq, rpa, rpb, q, dq, pa, pb, ca, cb, dummy, dummy)
        timer.block()
    sys.stdout.write("\n")
    # Return the last commands that have been sent to the valve.
    return ca, cb


def get_back_home_position(
    controller: CONTROLLER_T,
    q_home: np.ndarray,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    event_logger = get_event_logger()
    comm, ctrl, state = controller
    q0 = state.q
    ptp = PTP(q0, q_home, duration)
    qdes_func, dqdes_func = ptp.q, ptp.dq
    header_text = "Getting back to home position..."
    if event_logger:
        event_logger.debug(header_text)
        event_logger.debug("  duration: %s", duration)
        event_logger.debug("  q_home  : %s", q_home)
    ca, cb = control_position(controller, qdes_func, dqdes_func, duration, header_text=header_text)
    if event_logger:
        event_logger.debug("Done")
    return ca, cb


def get_back_home_pressure(
    controller: CONTROLLER_T,
    ca_home: np.ndarray,
    cb_home: np.ndarray,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    event_logger = get_event_logger()
    comm, ctrl, state = controller
    ca0, cb0 = np.zeros(ctrl.dof, dtype=float), np.zeros(ctrl.dof, dtype=float)
    ptp_ca = PTP(ca0, ca_home, duration, profile_name="const")
    ptp_cb = PTP(cb0, cb_home, duration, profile_name="const")
    ca_func, cb_func = ptp_ca.q, ptp_cb.q
    header_text = "Getting back to home position (by valve)..."
    if event_logger:
        event_logger.debug(header_text)
        event_logger.debug("  duration: %s", duration)
        event_logger.debug("  ca_home : %s", ca_home)
        event_logger.debug("  cb_home : %s", cb_home)
    ca, cb = control_pressure(controller, ca_func, cb_func, duration, header_text=header_text)
    if event_logger:
        event_logger.debug("Done")
    return ca, cb


class RobotInitializer:
    _dof: int
    _duration: float
    _manner: str
    _q_init: np.ndarray
    _ca_init: np.ndarray
    _cb_init: np.ndarray

    DEFAULT_DURATION = 5.0
    DEFAULT_MANNER = "position"
    DEFAULT_Q_INIT = 50.0
    DEFAULT_CA_INIT = 0.0
    DEFAULT_CB_INIT = 120.0

    def __init__(
        self,
        dof: int,
        *,
        config: str | Path | None = None,
        duration: float | None = None,
        manner: str | None = None,
        q_init: Sequence[float] | np.ndarray | float | None = None,
        ca_init: Sequence[float] | np.ndarray | float | None = None,
        cb_init: Sequence[float] | np.ndarray | float | None = None,
    ) -> None:
        self._dof = dof
        # Set default values
        self.duration = self.DEFAULT_DURATION
        self.set_manner(self.DEFAULT_MANNER)
        self.set_q_init(self.DEFAULT_Q_INIT)
        self.set_ca_init(self.DEFAULT_CA_INIT)
        self.set_cb_init(self.DEFAULT_CB_INIT)
        # Update values based on config file
        if config is not None:
            self.load_config(config)
        # Update values based on arguments
        self._update_values(duration, manner, q_init, ca_init, cb_init)

    @property
    def dof(self) -> int:
        return self._dof

    @property
    def duration(self) -> float:
        return self._duration

    @duration.setter
    def duration(self, duration: float) -> None:
        self._duration = duration

    def get_manner(self) -> str:
        return self._manner

    def set_manner(self, manner: str) -> str:
        match manner:
            case "position" | "pos" | "p" | "q":
                self._manner = "position"
            case "pressure" | "pres" | "pre" | "valve" | "v":
                self._manner = "pressure"
            case _:
                msg = f"Unrecognized manner for RobotInitializer: {manner}"
                raise ValueError(msg)
        return self._manner

    @staticmethod
    def normalize_array(dof: int, given_value: Sequence[float] | np.ndarray | float) -> np.ndarray:
        if isinstance(given_value, float | int):
            array = np.full((dof,), given_value, dtype=float)
        elif len(given_value) == dof:
            array = np.array(given_value, dtype=float)
        elif len(given_value) == 1:
            array = np.full((dof,), given_value[0], dtype=float)
        else:
            msg = f"Unable to set values due to size mismatch: dof={dof}, given_value={given_value}"
            raise ValueError(msg)
        return array

    def get_q_init(self) -> np.ndarray:
        return self._q_init

    def set_q_init(self, q_init: Sequence[float] | np.ndarray | float) -> np.ndarray:
        self._q_init = self.normalize_array(self.dof, q_init)
        return self._q_init

    def get_ca_init(self) -> np.ndarray:
        return self._ca_init

    def set_ca_init(self, ca_init: Sequence[float] | np.ndarray | float) -> np.ndarray:
        self._ca_init = self.normalize_array(self.dof, ca_init)
        return self._ca_init

    def get_cb_init(self) -> np.ndarray:
        return self._cb_init

    def set_cb_init(self, cb_init: Sequence[float] | np.ndarray | float) -> np.ndarray:
        self._cb_init = self.normalize_array(self.dof, cb_init)
        return self._cb_init

    def _update_values(
        self,
        duration: float | None,
        manner: str | None,
        q_init: Sequence[float] | np.ndarray | float | None,
        ca_init: Sequence[float] | np.ndarray | float | None,
        cb_init: Sequence[float] | np.ndarray | float | None,
    ) -> None:
        if duration is not None:
            self.duration = duration
        if manner is not None:
            self.set_manner(manner)
        if q_init is not None:
            self.set_q_init(q_init)
        if ca_init is not None:
            self.set_ca_init(ca_init)
        if cb_init is not None:
            self.set_cb_init(cb_init)

    def load_config(self, config: str | Path) -> None:
        with Path(config).open("rb") as f:
            c = tomllib.load(f)
        affetto_config = c["affetto"]
        init_config = affetto_config.get("init", None)
        if init_config is None:
            return
        duration = init_config.get("duration", None)
        manner = init_config.get("manner", None)
        q_init = init_config.get("q", None)
        ca_init = init_config.get("ca", None)
        cb_init = init_config.get("cb", None)
        self._update_values(duration, manner, q_init, ca_init, cb_init)

    def get_back_home(self, controller: CONTROLLER_T, duration: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        if duration is None:
            duration = self.duration
        if self.get_manner() == "pressure":
            ca, cb = get_back_home_pressure(controller, self.get_ca_init(), self.get_cb_init(), duration)
        else:
            ca, cb = get_back_home_position(controller, self.get_q_init(), duration)
        return ca, cb


class RandomTrajectory:
    active_joints: list[int]
    t0: float
    q0: np.ndarray
    update_t_range_list: list[tuple[float, float]]
    update_q_range_list: list[tuple[float, float]]
    update_q_limit_list: list[tuple[float, float]]
    update_profile: str
    async_update: bool
    rng: Generator
    sync_updater: PTP
    async_updater: list[PTP]

    def __init__(
        self,
        active_joints: list[int],
        t0: float,
        q0: np.ndarray,
        update_t_range: tuple[float, float] | list[tuple[float, float]],
        update_q_range: tuple[float, float] | list[tuple[float, float]],
        update_q_limit: tuple[float, float] | list[tuple[float, float]],
        update_profile: str = "trapezoidal",
        seed: int | GetGlobalRngT | None = GET_GLOBAL_RNG,
        *,
        async_update: bool = False,
    ) -> None:
        self.active_joints = active_joints
        self.t0 = t0
        self.q0 = q0.copy()
        self.set_update_t_range(active_joints, update_t_range)
        self.set_update_q_range(active_joints, update_q_range)
        self.set_update_q_limit(active_joints, update_q_limit)
        self.update_profile = update_profile
        self.async_update = async_update
        self.rng = get_rng(seed)
        self.reset_updater()

    @staticmethod
    def get_list_of_range(
        active_joints: list[int],
        given_range: tuple[float, float] | list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        if isinstance(given_range, tuple):
            x = (min(given_range), max(given_range))
            range_list = [x for _ in active_joints]
        elif len(active_joints) == len(given_range):
            range_list = [(min(x), max(x)) for x in given_range]
        else:
            msg = (
                "Lengths of lists (active joints / range list) are mismatch: "
                f"{len(active_joints)} vs {len(given_range)}"
            )
            raise ValueError(msg)
        return range_list

    def set_update_t_range(
        self,
        active_joints: list[int],
        update_t_range: tuple[float, float] | list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        self.update_t_range_list = self.get_list_of_range(active_joints, update_t_range)
        return self.update_t_range_list

    def set_update_q_range(
        self,
        active_joints: list[int],
        update_q_range: tuple[float, float] | list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        self.update_q_range_list = self.get_list_of_range(active_joints, update_q_range)
        return self.update_q_range_list

    def set_update_q_limit(
        self,
        active_joints: list[int],
        update_q_limit: tuple[float, float] | list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        self.update_q_limit_list = self.get_list_of_range(active_joints, update_q_limit)
        if isinstance(update_q_limit, tuple) and WAIST_JOINT_INDEX in active_joints:
            # When update_q_limit is given as tuple and waist joint is included in active
            # joints list, reduce limits of the waist joint.
            waist_index = active_joints.index(WAIST_JOINT_INDEX)
            self.update_q_limit_list[waist_index] = WAIST_JOINT_LIMIT
        return self.update_q_limit_list

    def generate_new_duration(self, t_range: tuple[float, float]) -> float:
        return self.rng.uniform(min(t_range), max(t_range))

    def generate_new_position(
        self,
        q0: float,
        q_range: tuple[float, float],
        q_limit: tuple[float, float],
        min_update_q_delta: float = MIN_UPDATE_Q_DELTA,
    ) -> float:
        dmin, dmax = min(q_range), max(q_range)
        qmin, qmax = min(q_limit), max(q_limit)
        qdes = q0
        ok = False
        while not ok:
            delta = self.rng.uniform(dmin, dmax)
            qdes = q0 + self.rng.choice([-1, 1]) * delta
            if qdes < qmin:
                qdes = qmin + (qmin - qdes)
            elif qdes > qmax:
                qdes = qmax - (qdes - qmax)
            qdes = max(min(qmax, qdes), qmin)
            qdiff = abs(qdes - q0)
            if qdiff > min_update_q_delta:
                ok = True
        return qdes

    def initialize_sync_updater(self, t0: float, q0: np.ndarray) -> PTP:
        if not all(x == self.update_t_range_list[0] for x in self.update_t_range_list):
            msg = "Enabled sync update but various update t range is given."
            warnings.warn(msg, stacklevel=2)

        active_q0 = q0[self.active_joints]
        duration = self.generate_new_duration(self.update_t_range_list[0])
        qdes = np.array(
            [
                self.generate_new_position(active_q0[i], self.update_q_range_list[i], self.update_q_limit_list[i])
                for i in range(len(self.active_joints))
            ],
        )
        return PTP(active_q0, qdes, duration, t0, profile_name=self.update_profile)

    def initialize_async_updater(self, t0: float, q0: np.ndarray) -> list[PTP]:
        ptp_list: list[PTP] = []
        for i, j in enumerate(self.active_joints):
            active_q0 = q0[j]
            duration = self.generate_new_duration(self.update_t_range_list[i])
            qdes = self.generate_new_position(active_q0, self.update_q_range_list[i], self.update_q_limit_list[i])
            ptp_list.append(PTP(active_q0, qdes, duration, t0, profile_name=self.update_profile))
        return ptp_list

    def initialize_updater(self, t0: float, q0: np.ndarray) -> None:
        if self.async_update:
            self.async_updater = self.initialize_async_updater(t0, q0)
        else:
            self.sync_updater = self.initialize_sync_updater(t0, q0)

    def update_sync_updater(self, t: float) -> None:
        ptp = self.sync_updater
        if ptp.t0 + ptp.T < t:
            new_t0 = ptp.t0 + ptp.T
            new_q0 = ptp.qF
            new_duration = self.generate_new_duration(self.update_t_range_list[0])
            new_qdes = np.array(
                [
                    self.generate_new_position(new_q0[i], self.update_q_range_list[i], self.update_q_limit_list[i])
                    for i in range(len(self.active_joints))
                ],
            )
            new_ptp = PTP(new_q0, new_qdes, new_duration, new_t0, profile_name=self.update_profile)
            self.sync_updater = new_ptp

    def update_async_updater(self, t: float) -> None:
        for i, ptp in enumerate(self.async_updater):
            if ptp.t0 + ptp.T < t:
                new_t0 = ptp.t0 + ptp.T
                new_q0 = ptp.qF
                new_duration = self.generate_new_duration(self.update_t_range_list[i])
                new_qdes = self.generate_new_position(new_q0, self.update_q_range_list[i], self.update_q_limit_list[i])
                new_ptp = PTP(new_q0, new_qdes, new_duration, new_t0, profile_name=self.update_profile)
                self.async_updater[i] = new_ptp

    def reset_updater(self, t0: float | None = None, q0: np.ndarray | None = None) -> None:
        if t0 is not None:
            self.t0 = t0
        else:
            t0 = self.t0

        if q0 is not None:
            self.q0 = q0
        else:
            q0 = self.q0

        self.initialize_updater(t0, q0)

    def get_qdes_func(self) -> Callable[[float], np.ndarray]:
        def qdes_async(t: float) -> np.ndarray:
            self.update_async_updater(t)
            qdes = self.q0.copy()
            qdes[self.active_joints] = [ptp.q(t) for ptp in self.async_updater]
            return qdes

        def qdes_sync(t: float) -> np.ndarray:
            self.update_sync_updater(t)
            qdes = self.q0.copy()
            qdes[self.active_joints] = self.sync_updater.q(t)
            return qdes

        if self.async_update:
            return qdes_async
        return qdes_sync

    def get_dqdes_func(self) -> Callable[[float], np.ndarray]:
        def dqdes_async(t: float) -> np.ndarray:
            self.update_async_updater(t)
            dqdes = np.zeros(self.q0.shape, dtype=float)
            dqdes[self.active_joints] = [ptp.dq(t) for ptp in self.async_updater]
            return dqdes

        def dqdes_sync(t: float) -> np.ndarray:
            self.update_sync_updater(t)
            dqdes = np.zeros(self.q0.shape, dtype=float)
            dqdes[self.active_joints] = self.sync_updater.dq(t)
            return dqdes

        if self.async_update:
            return dqdes_async
        return dqdes_sync


# Local Variables:
# jinx-local-words: "cb const dT dof dq dqdes init noqa pb pos qdes rb rdq rpa rpb rq"
# End:
