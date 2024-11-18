from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from affctrllib import AffPosCtrl

    from affetto_nn_ctrl._typing import Unknown


@dataclass
class DataAdapterParams:
    pass


Reference: TypeAlias = Callable[[float], np.ndarray]


class DataAdapter(metaclass=ABCMeta):
    _params: DataAdapterParams

    def __init__(self, params: DataAdapterParams) -> None:
        self._params = params

    @property
    def params(self) -> DataAdapterParams:
        return self._params

    @abstractmethod
    def to_feature(self, **data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def to_target(self, **data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def to_feature_for_predict(self, **data: np.ndarray | Reference) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def convert_output(self, y: np.ndarray, **data: np.ndarray) -> tuple[np.ndarray, ...]:
        raise NotImplementedError


class LinearRegressor(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> LinearRegressor: ...  # noqa: N803

    def predict(self, X: np.ndarray) -> np.ndarray | Unknown: ...  # noqa: N803

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> float: ...  # noqa: N803


class MultiLayerPerceptronRegressor(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> MultiLayerPerceptronRegressor: ...  # noqa: N803

    def predict(self, X: np.ndarray) -> np.ndarray | Unknown: ...  # noqa: N803

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> float | Unknown: ...  # noqa: N803


Regressor: TypeAlias = LinearRegressor | MultiLayerPerceptronRegressor


class CtrlAdapter:
    ctrl: AffPosCtrl
    model: Regressor
    data_adapter: DataAdapter
    _updater: Callable[
        [float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Reference, Reference],
        tuple[np.ndarray, np.ndarray],
    ]

    def __init__(self, ctrl: AffPosCtrl, model: Regressor | None, data_adapter: DataAdapter) -> None:
        self.ctrl = ctrl
        if model is None:
            self._updater = self.update_ctrl
        else:
            self._updater = self.update_model
        self.data_adapter = data_adapter

    def update_ctrl(
        self,
        t: float,
        q: np.ndarray,
        dq: np.ndarray,
        pa: np.ndarray,
        pb: np.ndarray,
        qdes: Reference,
        dqdes: Reference,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.ctrl.update(t, q, dq, pa, pb, qdes(t), dqdes(t))

    def update_model(
        self,
        t: float,
        q: np.ndarray,
        dq: np.ndarray,
        pa: np.ndarray,
        pb: np.ndarray,
        qdes: Reference,
        dqdes: Reference,
    ) -> tuple[np.ndarray, np.ndarray]:
        ca, cb = self.ctrl.update(t, q, dq, pa, pb, qdes(t), dqdes(t))
        x = self.data_adapter.to_feature_for_predict(q=q, dq=dq, pa=pa, pb=pb, qdes=qdes, dqdes=dqdes)
        y = np.ravel(self.model.predict(np.atleast_2d(x)))
        ca, cb = self.data_adapter.convert_output(y, ca=ca, cb=cb)
        return ca, cb

    def update(
        self,
        t: float,
        q: np.ndarray,
        dq: np.ndarray,
        pa: np.ndarray,
        pb: np.ndarray,
        qdes: Reference,
        dqdes: Reference,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._updater(t, q, dq, pa, pb, qdes, dqdes)


# Local Variables:
# jinx-local-words: "noqa"
# End:
