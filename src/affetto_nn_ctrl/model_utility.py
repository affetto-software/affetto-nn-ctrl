from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Protocol, TypeAlias, TypedDict, TypeVar, cast, overload

import numpy as np
from pyplotutil.datautil import Data

if TYPE_CHECKING:
    import pandas as pd
    from affctrllib import AffPosCtrl

    from affetto_nn_ctrl._typing import Unknown

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired


@dataclass
class DataAdapterParamsBase:
    pass


class StatesBase(TypedDict):
    pass


class RefsBase(TypedDict):
    pass


class InputsBase(TypedDict):
    pass


DataAdapterParamsType = TypeVar("DataAdapterParamsType", bound=DataAdapterParamsBase)
StatesType = TypeVar("StatesType", bound=StatesBase)
RefsType = TypeVar("RefsType", bound=RefsBase)
InputsType = TypeVar("InputsType", bound=InputsBase)
Reference: TypeAlias = Callable[[float], np.ndarray]


class DefaultStates(StatesBase):
    q: np.ndarray
    dq: np.ndarray
    ddq: NotRequired[np.ndarray]
    pa: np.ndarray
    pb: np.ndarray


class DefaultRefs(RefsBase):
    qdes: Reference
    dqdes: NotRequired[Reference]


class DefaultInputs(InputsBase):
    ca: np.ndarray
    cb: np.ndarray


class DataAdapterBase(ABC, Generic[DataAdapterParamsType, StatesType, RefsType, InputsType]):
    _params: DataAdapterParamsType

    def __init__(self, params: DataAdapterParamsType) -> None:
        self._params = params

    @property
    def params(self) -> DataAdapterParamsType:
        return self._params

    @abstractmethod
    def make_feature(self, dataset: Data) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def make_target(self, dataset: Data) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def make_model_input(self, t: float, states: StatesType, refs: RefsType) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def make_ctrl_input(self, y: np.ndarray, base_inputs: InputsType) -> tuple[np.ndarray, ...]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
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


def extract_data(
    dataset: Data,
    keys: Iterable[str],
    shift: int,
    keys_replace: Iterable[str] | None = None,
) -> pd.DataFrame:
    subset = dataset.df.loc[:, keys]
    extracted = subset[:shift] if shift < 0 else subset[shift:]
    if keys_replace is not None:
        extracted = extracted.rename(columns=dict(zip(keys, keys_replace, strict=True)))
    return extracted.reset_index(drop=True)


def load_train_datasets(
    train_datasets: Data | Iterable[Data],
    adapter: DataAdapterBase,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(train_datasets, Data):
        train_datasets = [train_datasets]
    x_train = cast(np.ndarray, None)
    y_train = cast(np.ndarray, None)
    for dataset in train_datasets:
        _x_train = adapter.make_feature(dataset)
        _y_train = adapter.make_target(dataset)
        x_train = np.vstack((x_train, _x_train)) if x_train else np.copy(_x_train)
        y_train = np.vstack((y_train, _y_train)) if y_train else np.copy(_y_train)
    if len(y_train.shape) == 2 and y_train.shape[1] == 1:  # noqa: PLR2004
        y_train = np.ravel(y_train)
    return x_train, y_train


@overload
def train_model(
    model: Regressor,
    x_train_or_datasets: np.ndarray,
    y_train_or_adapter: np.ndarray,
) -> Regressor: ...


@overload
def train_model(
    model: Regressor,
    x_train_or_datasets: Data | Iterable[Data],
    y_train_or_adapter: DataAdapterBase,
) -> Regressor: ...


def train_model(model, x_train_or_datasets, y_train_or_adapter) -> Regressor:
    if isinstance(x_train_or_datasets, np.ndarray) and isinstance(y_train_or_adapter, np.ndarray):
        x_train = x_train_or_datasets
        y_train = y_train_or_adapter
    else:
        x_train, y_train = load_train_datasets(x_train_or_datasets, y_train_or_adapter)
    return model.fit(x_train, y_train)


CtrlAdapterUpdater: TypeAlias = Callable[
    [float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Reference, Reference],
    tuple[np.ndarray, np.ndarray],
]


class CtrlAdapter(Generic[DataAdapterParamsType, StatesType, RefsType, InputsType]):
    ctrl: AffPosCtrl
    model: Regressor
    data_adapter: DataAdapterBase[DataAdapterParamsType, StatesType, RefsType, InputsType]
    _updater: CtrlAdapterUpdater

    def __init__(
        self,
        ctrl: AffPosCtrl,
        model: Regressor | None,
        data_adapter: DataAdapterBase[DataAdapterParamsType, StatesType, RefsType, InputsType],
    ) -> None:
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


class DefaultCtrlAdapter(CtrlAdapter[DataAdapterParamsType, DefaultStates, DefaultRefs, DefaultInputs]):
    def __init__(
        self,
        ctrl: AffPosCtrl,
        model: Regressor | None,
        data_adapter: DataAdapterBase[DataAdapterParamsType, DefaultStates, DefaultRefs, DefaultInputs],
    ) -> None:
        super().__init__(ctrl, model, data_adapter)

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
        x = self.data_adapter.make_model_input(
            t,
            {"q": q, "dq": dq, "pa": pa, "pb": pb},
            {"qdes": qdes, "dqdes": dqdes},
        )
        y = self.model.predict(x)
        ca, cb = self.data_adapter.make_ctrl_input(y, {"ca": ca, "cb": cb})
        return ca, cb


# Local Variables:
# jinx-local-words: "Params cb dq dqdes noqa npqa pb qdes"
# End:
