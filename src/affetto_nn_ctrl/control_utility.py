from __future__ import annotations

import os
import random
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from affctrllib import PTP, AffComm, AffPosCtrl, AffStateThread, Logger, Timer

if TYPE_CHECKING:
    from typing import Callable, Iterable


def set_seed(seed: int | None) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002


T = TypeVar("T")


def prepare_ctrl(
    config: str,
    srate: float | None,
    crate: float | None,
    waiting_time: float = 5.0,
) -> tuple[AffComm, AffPosCtrl, AffStateThread]:
    comm = AffComm(config_path=config)
    comm.create_command_socket()
    state = AffStateThread(config=config, freq=srate, logging=False, output=None, butterworth=True)
    ctrl = AffPosCtrl(config_path=config, freq=crate)
    state.prepare()
    state.start()
    print("Waiting until robot gets stationary...")  # noqa: T201
    time.sleep(waiting_time)
    return comm, ctrl, state


def get_datetime(_format: str = "%Y%m%dT%H%M%S") -> str:
    return datetime.now().strftime(_format)  # noqa: DTZ005


def prepare_output_directory(output_directory: str | Path | None, make_backup: bool = False) -> Path | None:
    if output_directory is None:
        warnings.warn(f"Output directory is not provdided. No data will be saved.")
        return

    path = Path(output_directory)
    if make_backup:
        symlink = path / "latest"
        path = path / get_datetime()
        if not symlink.exists() or symlink.is_symlink():
            try:
                os.remove(symlink)
            except OSError:
                pass
            finally:
                os.symlink(path.absolute(), symlink)
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_default_logger(dof: int) -> Logger:
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
        dq = np.zeros((len(q0),))
        return dq

    return qdes_func, dqdes_func


def control_position(
    comm: AffComm,
    ctrl: AffPosCtrl,
    state: AffStateThread,
    qdes_func: Callable[[float], np.ndarray | float],
    dqdes_func: Callable[[float], np.ndarray | float],
    duration: float,
    logger: Logger | None = None,
    header_text: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    timer = Timer(rate=ctrl.freq)
    if logger is not None:
        logger.erase_data()
    timer.start()
    t = 0
    ca, cb = np.zeros((ctrl.dof,)), np.zeros((ctrl.dof,))
    while t < duration:
        print(f"\r{header_text} [{t:6.2f}/{duration:.2f}]", end="")  # noqa: T201
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
    # return the last command sent to the valve
    print("")  # noqa: T201
    return ca, cb


def getbackhome_position(
    comm: AffComm,
    ctrl: AffPosCtrl,
    state: AffStateThread,
    qhome: np.ndarray,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    q0 = state.q
    ptp = PTP(q0, qhome, duration)
    qdes_func, dqdes_func = ptp.q, ptp.dq
    ca, cb = control_position(
        comm,
        ctrl,
        state,
        qdes_func,
        dqdes_func,
        duration,
        header_text=f"Getting back to home position...",
    )
    return ca, cb


def control_valve(
    comm: AffComm,
    ctrl: AffPosCtrl,
    state: AffStateThread,
    ca_func: Callable[[float], np.ndarray | float],
    cb_func: Callable[[float], np.ndarray | float],
    duration: float,
    logger: Logger | None = None,
    header_text: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    timer = Timer(rate=ctrl.freq)
    if logger is not None:
        logger.erase_data()
    dummy = np.asarray([-1.0 for _ in range(ctrl.dof)])

    timer.start()
    t = 0
    ca, cb = np.zeros((ctrl.dof,)), np.zeros((ctrl.dof,))
    while t < duration:
        print(f"\r{header_text} [{t:6.2f}/{duration:.2f}]", end="")  # noqa: T201
        t = timer.elapsed_time()
        rq, rdq, rpa, rpb = state.get_raw_states()
        q, dq, pa, pb = state.get_states()
        ca = np.asarray(ca_func(t))
        cb = np.asarray(cb_func(t))
        comm.send_commands(ca, cb)
        if logger is not None:
            logger.store(t, rq, rdq, rpa, rpb, q, dq, pa, pb, ca, cb, dummy, dummy)
        timer.block()
    # return the last command sent to the valve
    print("")  # noqa: T201
    return ca, cb


def getbackhome_valve(
    comm: AffComm,
    ctrl: AffPosCtrl,
    state: AffStateThread,
    cahome: np.ndarray,
    cbhome: np.ndarray,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    ca0, cb0 = np.zeros((ctrl.dof,)), np.zeros((ctrl.dof,))
    # ca0, cb0 = state.pa, state.pb
    ptp_ca = PTP(ca0, cahome, duration, profile_name="const")
    ptp_cb = PTP(cb0, cbhome, duration, profile_name="const")
    ca_func, cb_func = ptp_ca.q, ptp_cb.q
    ca, cb = control_valve(
        comm,
        ctrl,
        state,
        ca_func,
        cb_func,
        duration,
        header_text=f"Getting back to home position (by valve)...",
    )
    return ca, cb


def control_command(
    comm: AffComm,
    ctrl: AffPosCtrl,
    state: AffStateThread,
    commands: CommandSet,
    duration: float,
    seed: int | None,
    logger: Logger | None = None,
    header_text: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    timer = Timer(rate=ctrl.freq)
    if logger is not None:
        logger.erase_data()
    dummy = np.asarray([-1.0 for _ in range(ctrl.dof)])

    # reset commands
    set_seed(seed)
    commands.reset()

    t = 0
    cnt = 0
    ca, cb = np.zeros((ctrl.dof,)), np.zeros((ctrl.dof,))
    timer.start()
    while t < duration:
        print(f"\r{header_text} [{t:6.2f}/{duration:.2f}]", end="")  # noqa: T201
        # t = timer.elapsed_time()
        t = timer.period * cnt
        rq, rdq, rpa, rpb = state.get_raw_states()
        q, dq, pa, pb = state.get_states()
        ca, cb = commands.update(t, q, dq)
        # ca, cb = commands.update(t, rq, rdq)
        comm.send_commands(ca, cb)
        if logger is not None:
            logger.store(t, rq, rdq, rpa, rpb, q, dq, pa, pb, ca, cb, dummy, dummy)
        timer.block()
        cnt += 1
    print("")  # noqa: T201
    # return the last command sent to the valve
    return ca, cb


class RandomTrajectory:
    joints: list[int]
    q0: np.ndarray
    t0: float
    update_t_range: tuple[float, float]
    update_q_range: tuple[float, float]
    q_limit: tuple[float, float]
    q_limits: list[tuple[float, float]]
    ptp_list: list[PTP]
    profile: str

    def __init__(
        self,
        joints: list[int],
        q0: np.ndarray,
        t0: float,
        update_t_range: tuple[float, float],
        update_q_range: tuple[float, float],
        q_limit: tuple[float, float],
        profile: str = "trapezoidal",
        seed: int | None = None,
    ) -> None:
        self.joints = joints
        self.q0 = q0.copy()
        self.t0 = t0
        self.update_t_range = update_t_range
        self.update_q_range = update_q_range
        self.q_limit = q_limit
        self.q_limits = [self.q_limit for _ in self.joints]
        for i, j in enumerate(self.joints):
            if j == 0:
                # Reduce limits of waist joint.
                self.q_limits[i] = (40.0, 60.0)
        self.profile = profile
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.ptp_list = self.initialize_ptp()

    def _get_new_T_qdes(
        self,
        q0: float,
        t_range: tuple[float, float],
        q_range: tuple[float, float],
        q_limit: tuple[float, float],
    ) -> tuple[float, float]:
        qmin, qmax = min(q_limit), max(q_limit)
        qdes = q0
        T = random.uniform(min(t_range), max(t_range))
        ok = False
        while not ok:
            q_diff = random.uniform(min(q_range), max(q_range))
            qdes = random.choice([-1, 1]) * q_diff + q0
            if qdes < qmin:
                qdes = qmin + (qmin - qdes)
            elif qdes > qmax:
                qdes = qmax - (qdes - qmax)
            qdes = max(min(qmax, qdes), qmin)
            if abs(qdes - q0) > 0.0001:
                ok = True
        return T, qdes

    def initialize_ptp(self) -> list[PTP]:
        ptp_list: list[PTP] = []
        for i in self.joints:
            T, qdes = self._get_new_T_qdes(self.q0[i], self.update_t_range, self.update_q_range, self.q_limits[i])
            ptp_list.append(PTP(self.q0[i], qdes, T, self.t0, profile_name=self.profile))
        return ptp_list

    def update_ptp(self, t: float) -> None:
        for i, ptp in enumerate(self.ptp_list):
            if ptp.t0 + ptp.T < t:
                new_t0 = ptp.t0 + ptp.T
                new_q0 = ptp.qF
                new_T, new_qdes = self._get_new_T_qdes(
                    new_q0,
                    self.update_t_range,
                    self.update_q_range,
                    self.q_limits[i],
                )
                new_ptp = PTP(new_q0, new_qdes, new_T, new_t0, profile_name=self.profile)
                self.ptp_list[i] = new_ptp

    def qdes(self, t: float) -> np.ndarray:
        self.update_ptp(t)
        qdes = self.q0.copy()
        qdes[self.joints] = [ptp.q(t) for ptp in self.ptp_list]
        return qdes

    def dqdes(self, t: float) -> np.ndarray:
        self.update_ptp(t)
        dqdes = np.zeros(self.q0.shape)
        dqdes[self.joints] = [ptp.dq(t) for ptp in self.ptp_list]
        return dqdes


# Local Variables:
# jinx-local-words: "cb dT dq dqdes noqa pb qdes rdq rpa rpb rq"
# End:
