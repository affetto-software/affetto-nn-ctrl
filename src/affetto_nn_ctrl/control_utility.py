# ruff: noqa: S311
from __future__ import annotations

import sys
import time
import warnings
from typing import TYPE_CHECKING

import numpy as np
from affctrllib import PTP, AffComm, AffPosCtrl, AffStateThread, Logger, Timer
from numpy.random import Generator, default_rng

from affetto_nn_ctrl.event_logging import get_event_logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


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
) -> tuple[AffComm, AffPosCtrl, AffStateThread]:
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


def _reset_logger(logger: Logger | None, log_filename: str | Path | None) -> Logger | None:
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


def control_position(
    comm: AffComm,
    ctrl: AffPosCtrl,
    state: AffStateThread,
    qdes_func: Callable[[float], np.ndarray | float],
    dqdes_func: Callable[[float], np.ndarray | float],
    duration: float,
    logger: Logger | None = None,
    log_filename: str | Path | None = None,
    header_text: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    _reset_logger(logger, log_filename)
    ca, cb = np.zeros(ctrl.dof, dtype=float), np.zeros(ctrl.dof, dtype=float)
    timer = Timer(rate=ctrl.freq)

    timer.start()
    t = 0.0
    while t < duration:
        sys.stdout.write(f"\r{header_text} [{t:6.2f}/{duration:.2f}]")
        t = timer.elapsed_time()
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


def control_valve(
    comm: AffComm,
    ctrl: AffPosCtrl,
    state: AffStateThread,
    ca_func: Callable[[float], np.ndarray | float],
    cb_func: Callable[[float], np.ndarray | float],
    duration: float,
    logger: Logger | None = None,
    log_filename: str | Path | None = None,
    header_text: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    _reset_logger(logger, log_filename)
    ca, cb = np.zeros(ctrl.dof, dtype=float), np.zeros(ctrl.dof, dtype=float)
    timer = Timer(rate=ctrl.freq)
    dummy = np.asarray([-1.0 for _ in range(ctrl.dof)])

    timer.start()
    t = 0.0
    while t < duration:
        sys.stdout.write(f"\r{header_text} [{t:6.2f}/{duration:.2f}]")
        t = timer.elapsed_time()
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
    comm: AffComm,
    ctrl: AffPosCtrl,
    state: AffStateThread,
    q_home: np.ndarray,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    event_logger = get_event_logger()
    q0 = state.q
    ptp = PTP(q0, q_home, duration)
    qdes_func, dqdes_func = ptp.q, ptp.dq
    msg = "Getting back to home position..."
    if event_logger:
        event_logger.debug(msg)
    ca, cb = control_position(comm, ctrl, state, qdes_func, dqdes_func, duration, header_text=msg)
    if event_logger:
        event_logger.debug("Done")
    return ca, cb


def get_back_home_valve(
    comm: AffComm,
    ctrl: AffPosCtrl,
    state: AffStateThread,
    ca_home: np.ndarray,
    cb_home: np.ndarray,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    event_logger = get_event_logger()
    ca0, cb0 = np.zeros(ctrl.dof, dtype=float), np.zeros(ctrl.dof, dtype=float)
    ptp_ca = PTP(ca0, ca_home, duration, profile_name="const")
    ptp_cb = PTP(cb0, cb_home, duration, profile_name="const")
    ca_func, cb_func = ptp_ca.q, ptp_cb.q
    msg = "Getting back to home position (by valve)..."
    if event_logger:
        event_logger.debug(msg)
    ca, cb = control_valve(comm, ctrl, state, ca_func, cb_func, duration, header_text=msg)
    if event_logger:
        event_logger.debug("Done")
    return ca, cb


class RandomTrajectory:
    active_joints: list[int]
    q0: np.ndarray
    t0: float
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
        q0: np.ndarray,
        t0: float,
        update_t_range: tuple[float, float] | list[tuple[float, float]],
        update_q_range: tuple[float, float] | list[tuple[float, float]],
        update_q_limit: tuple[float, float] | list[tuple[float, float]],
        update_profile: str = "trapezoidal",
        seed: int | GetGlobalRngT | None = GET_GLOBAL_RNG,
        *,
        async_update: bool = False,
    ) -> None:
        self.active_joints = active_joints
        self.q0 = q0.copy()
        self.t0 = t0
        self.set_update_t_range(active_joints, update_t_range)
        self.set_update_q_range(active_joints, update_q_range)
        self.set_update_q_limit(active_joints, update_q_limit)
        self.update_profile = update_profile
        self.async_update = async_update
        self.rng = get_rng(seed)
        self.initialize_updater()

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

    def initialize_sync_updater(self) -> PTP:
        if not all(x == self.update_t_range_list[0] for x in self.update_t_range_list):
            msg = "Enabled sync update but various update t range is given."
            warnings.warn(msg, stacklevel=2)

        q0 = self.q0[self.active_joints]
        duration = self.generate_new_duration(self.update_t_range_list[0])
        qdes = np.array(
            [
                self.generate_new_position(q0[i], self.update_q_range_list[i], self.update_q_limit_list[i])
                for i in range(len(self.active_joints))
            ],
        )
        return PTP(q0, qdes, duration, self.t0, profile_name=self.update_profile)

    def initialize_async_updater(self) -> list[PTP]:
        ptp_list: list[PTP] = []
        for i, j in enumerate(self.active_joints):
            q0 = self.q0[j]
            duration = self.generate_new_duration(self.update_t_range_list[i])
            qdes = self.generate_new_position(q0, self.update_q_range_list[i], self.update_q_limit_list[i])
            ptp_list.append(PTP(q0, qdes, duration, self.t0, profile_name=self.update_profile))
        return ptp_list

    def initialize_updater(self) -> None:
        if self.async_update:
            self.async_updater = self.initialize_async_updater()
        else:
            self.sync_updater = self.initialize_sync_updater()

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
# jinx-local-words: "cb const dT dq dqdes noqa pb qdes rdq rpa rpb rq"
# End:
