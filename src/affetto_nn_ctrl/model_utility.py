from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Protocol, TypeAlias, TypedDict, TypeVar, cast, overload

import joblib
import numpy as np
from affctrllib import Logger, Timer
from pyplotutil.datautil import Data
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

from affetto_nn_ctrl.control_utility import reset_logger, select_time_updater
from affetto_nn_ctrl.event_logging import event_logger

if TYPE_CHECKING:
    import pandas as pd
    from affctrllib import AffPosCtrl

    from affetto_nn_ctrl import CONTROLLER_T, RefFuncType
    from affetto_nn_ctrl._typing import T, Unknown

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


class DefaultStates(StatesBase):
    q: np.ndarray
    dq: np.ndarray
    ddq: NotRequired[np.ndarray]
    pa: np.ndarray
    pb: np.ndarray


class DefaultRefs(RefsBase):
    qdes: np.ndarray
    dqdes: NotRequired[np.ndarray]


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


class DummyDataAdapter(DataAdapterBase[DataAdapterParamsBase, StatesBase, RefsBase, InputsBase]):
    def make_feature(self, dataset: Data) -> np.ndarray:
        raise NotImplementedError

    def make_target(self, dataset: Data) -> np.ndarray:
        raise NotImplementedError

    def make_model_input(self, t: float, states: StatesBase, refs: RefsBase) -> np.ndarray:
        raise NotImplementedError

    def make_ctrl_input(self, y: np.ndarray, base_inputs: InputsBase) -> tuple[np.ndarray, ...]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


dummy_data_adapter = DummyDataAdapter(DataAdapterParamsBase())


@dataclass
class PreviewRefParams(DataAdapterParamsBase):
    active_joints: list[int]
    ctrl_step: int = 1
    preview_step: int = 0


class PreviewRef(DataAdapterBase[PreviewRefParams, DefaultStates, DefaultRefs, DefaultInputs]):
    def __init__(self, params: PreviewRefParams) -> None:
        super().__init__(params)

    def make_feature(self, dataset: Data) -> np.ndarray:
        raise NotImplementedError

    def make_target(self, dataset: Data) -> np.ndarray:
        raise NotImplementedError

    def make_model_input(self, t: float, states: StatesBase, refs: RefsBase) -> np.ndarray:
        raise NotImplementedError

    def make_ctrl_input(self, y: np.ndarray, base_inputs: InputsBase) -> tuple[np.ndarray, ...]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


@dataclass
class DelayStatesParams(DataAdapterParamsBase):
    active_joints: list[int]
    ctrl_step: int = 1
    delay_step: int = 0


class DelayStates(DataAdapterBase[DelayStatesParams, DefaultStates, DefaultRefs, DefaultInputs]):
    def __init__(self, params: DelayStatesParams) -> None:
        super().__init__(params)

    def make_feature(self, dataset: Data) -> np.ndarray:
        raise NotImplementedError

    def make_target(self, dataset: Data) -> np.ndarray:
        raise NotImplementedError

    def make_model_input(self, t: float, states: StatesBase, refs: RefsBase) -> np.ndarray:
        raise NotImplementedError

    def make_ctrl_input(self, y: np.ndarray, base_inputs: InputsBase) -> tuple[np.ndarray, ...]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


@dataclass
class DelayStatesAllParams(DataAdapterParamsBase):
    active_joints: list[int]
    ctrl_step: int = 1
    delay_step: int = 0


class DelayStatesAll(DataAdapterBase[DelayStatesAllParams, DefaultStates, DefaultRefs, DefaultInputs]):
    def __init__(self, params: DelayStatesAllParams) -> None:
        super().__init__(params)

    def make_feature(self, dataset: Data) -> np.ndarray:
        raise NotImplementedError

    def make_target(self, dataset: Data) -> np.ndarray:
        raise NotImplementedError

    def make_model_input(self, t: float, states: StatesBase, refs: RefsBase) -> np.ndarray:
        raise NotImplementedError

    def make_ctrl_input(self, y: np.ndarray, base_inputs: InputsBase) -> tuple[np.ndarray, ...]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class Scaler(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Scaler: ...  # noqa: N803

    def inverse_transform(self, X: np.ndarray) -> np.ndarray: ...  # noqa: N803

    def transform(self, X: np.ndarray) -> np.ndarray | Unknown: ...  # noqa: N803

    def get_params(self) -> dict[str, object]: ...


class Regressor(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Regressor: ...  # noqa: N803

    def predict(self, X: np.ndarray) -> np.ndarray | Unknown: ...  # noqa: N803

    def score(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float | Unknown: ...

    def get_params(self) -> dict[str, object]: ...


DATA_ADAPTER_MAP: Mapping[str, tuple[type[DataAdapterBase], type[DataAdapterParamsBase]]] = {
    "preview-ref": (PreviewRef, PreviewRefParams),
    "delay-states": (DelayStates, DelayStatesParams),
    "delay-states-all": (DelayStatesAll, DelayStatesAllParams),
}

SCALER_MAP: Mapping[str, tuple[type[Scaler], None]] = {
    "std": (StandardScaler, None),
    "minmax": (MinMaxScaler, None),
    "maxabs": (MaxAbsScaler, None),
    "robust": (RobustScaler, None),
}


REGRESSOR_MAP: Mapping[str, tuple[type[Regressor], None]] = {
    "linear": (LinearRegression, None),
    "ridge": (Ridge, None),
    "mlp": (MLPRegressor, None),
}


def pop_multi_keys(config: dict[str, Unknown], keys: Iterable[str]) -> dict[str, Unknown]:
    return {key: config.pop(key, {}) for key in keys}


def _load_from_map(
    config: dict[str, Unknown],
    _map: Mapping[str, tuple[type[T], type[DataAdapterParamsBase] | None]],
    _display: str,
) -> T:
    try:
        name = config.pop("name")
    except KeyError as e:
        msg = f"'name' is required to load {_display}"
        raise KeyError(msg) from e

    try:
        _type, params_type = _map[name]
    except KeyError as e:
        msg = f"unknown {_display} name: {name}"
        raise KeyError(msg) from e

    params_set = pop_multi_keys(config, _map.keys())
    selected_params_set = config.pop("params", None)
    if isinstance(selected_params_set, str):
        try:
            config.update(params_set[name][selected_params_set])
        except KeyError as e:
            msg = f"unknown parameter set name: {selected_params_set}"
            raise KeyError(msg) from e
    elif selected_params_set is not None:
        msg = f"value of 'params' is not string: {selected_params_set}"
        raise ValueError(msg)

    if params_type is None:
        return _type(**config)
    return _type(params_type(**config))  # type: ignore[call-arg]


def update_config_by_selector(
    config: dict[str, Unknown],
    selector: str | None,
) -> dict[str, Unknown]:
    if selector is not None:
        splitted = selector.split(".")
        config.update(name=splitted[0])
        if len(splitted) > 1:
            config.update(params=splitted[1])
    return config


def load_data_adapter(config: dict[str, Unknown], active_joints: list[int] | None = None) -> DataAdapterBase:
    adapter = _load_from_map(config, DATA_ADAPTER_MAP, "data adapter")
    if active_joints is not None:
        adapter.params.active_joints = active_joints
    return adapter


def load_scaler(config: dict[str, Unknown]) -> Scaler:
    return _load_from_map(config, SCALER_MAP, "scaler")


def load_regressor(config: dict[str, Unknown]) -> Regressor:
    return _load_from_map(config, REGRESSOR_MAP, "regressor")


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
    adapter: DataAdapterBase[DataAdapterParamsType, StatesType, RefsType, InputsType],
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(train_datasets, Data):
        train_datasets = [train_datasets]
    x_train = cast(np.ndarray, None)
    y_train = cast(np.ndarray, None)
    for dataset in train_datasets:
        _x_train = adapter.make_feature(dataset)
        _y_train = adapter.make_target(dataset)
        x_train = np.vstack((x_train, _x_train)) if x_train is not None else np.copy(_x_train)
        y_train = np.vstack((y_train, _y_train)) if y_train is not None else np.copy(_y_train)
        if dataset.is_loaded_from_file():
            event_logger().info("Loaded dataset: %s", dataset.datapath)
    if x_train is None or y_train is None:
        msg = f"No data sets found: {train_datasets}"
        raise RuntimeError(msg)
    if len(y_train.shape) == 2 and y_train.shape[1] == 1:  # noqa: PLR2004
        y_train = np.ravel(y_train)
    return x_train, y_train


def construct_model(*steps: Scaler | Regressor) -> Regressor | Pipeline:
    return make_pipeline(*steps)


TrainableModel = TypeVar("TrainableModel", Regressor, Pipeline)


@overload
def train_model(
    model: TrainableModel,
    x_train_or_datasets: np.ndarray,
    y_train_or_adapter: np.ndarray,
) -> TrainableModel: ...


@overload
def train_model(
    model: TrainableModel,
    x_train_or_datasets: Data | Iterable[Data],
    y_train_or_adapter: DataAdapterBase[DataAdapterParamsType, StatesType, RefsType, InputsType],
) -> TrainableModel: ...


def train_model(model, x_train_or_datasets, y_train_or_adapter):
    if isinstance(x_train_or_datasets, np.ndarray) and isinstance(y_train_or_adapter, np.ndarray):
        x_train = x_train_or_datasets
        y_train = y_train_or_adapter
    else:
        x_train, y_train = load_train_datasets(x_train_or_datasets, y_train_or_adapter)
    event_logger().debug("x_train.shape = %s", x_train.shape)
    event_logger().debug("y_train.shape = %s", y_train.shape)
    return model.fit(x_train, y_train)


@dataclass
class TrainedModel(Generic[DataAdapterParamsType, StatesType, RefsType, InputsType]):
    model: Regressor | Pipeline
    adapter: DataAdapterBase[DataAdapterParamsType, StatesType, RefsType, InputsType]

    def get_params(self) -> dict:
        return self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray | Unknown:  # noqa: N803
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> float | Unknown:  # noqa: N803
        return self.model.score(X, y, sample_weight)

    @property
    def model_name(self) -> str:
        if isinstance(self.model, Pipeline):
            return " -> ".join(str(step[1]) for step in self.model.steps)
        return str(self.model)


def dump_model(model: TrainedModel, output: str | Path) -> Path:
    joblib.dump(model, output)
    return Path(output)


def load_model(model_filepath: str | Path) -> TrainedModel:
    return joblib.load(model_filepath)


CtrlAdapterUpdater: TypeAlias = Callable[
    [float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]


class CtrlAdapter(Generic[DataAdapterParamsType, StatesType, RefsType, InputsType]):
    ctrl: AffPosCtrl
    model: TrainedModel[DataAdapterParamsType, StatesType, RefsType, InputsType]
    _updater: CtrlAdapterUpdater

    def __init__(
        self,
        ctrl: AffPosCtrl,
        model: TrainedModel[DataAdapterParamsType, StatesType, RefsType, InputsType] | None,
    ) -> None:
        self.ctrl = ctrl
        if model is None:
            self._updater = self.update_ctrl
        else:
            self.model = model
            self._updater = self.update_model

    def update_ctrl(
        self,
        t: float,
        q: np.ndarray,
        dq: np.ndarray,
        pa: np.ndarray,
        pb: np.ndarray,
        qdes: np.ndarray,
        dqdes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.ctrl.update(t, q, dq, pa, pb, qdes, dqdes)

    def update_model(
        self,
        t: float,
        q: np.ndarray,
        dq: np.ndarray,
        pa: np.ndarray,
        pb: np.ndarray,
        qdes: np.ndarray,
        dqdes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ca, cb = self.ctrl.update(t, q, dq, pa, pb, qdes, dqdes)
        return ca, cb

    def update(
        self,
        t: float,
        q: np.ndarray,
        dq: np.ndarray,
        pa: np.ndarray,
        pb: np.ndarray,
        qdes: np.ndarray,
        dqdes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._updater(t, q, dq, pa, pb, qdes, dqdes)


class DefaultCtrlAdapter(CtrlAdapter[DataAdapterParamsType, DefaultStates, DefaultRefs, DefaultInputs]):
    def __init__(
        self,
        ctrl: AffPosCtrl,
        model: TrainedModel[DataAdapterParamsType, DefaultStates, DefaultRefs, DefaultInputs] | None,
    ) -> None:
        super().__init__(ctrl, model)

    def update_model(
        self,
        t: float,
        q: np.ndarray,
        dq: np.ndarray,
        pa: np.ndarray,
        pb: np.ndarray,
        qdes: np.ndarray,
        dqdes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ca, cb = self.ctrl.update(t, q, dq, pa, pb, qdes, dqdes)
        x = self.model.adapter.make_model_input(
            t,
            {"q": q, "dq": dq, "pa": pa, "pb": pb},
            {"qdes": qdes, "dqdes": dqdes},
        )
        y = self.model.predict(x)
        ca, cb = self.model.adapter.make_ctrl_input(y, {"ca": ca, "cb": cb})
        return ca, cb


DefaultCtrlAdapterType: TypeAlias = DefaultCtrlAdapter[DataAdapterParamsBase]
DefaultTrainedModelType: TypeAlias = TrainedModel[DataAdapterParamsBase, DefaultStates, DefaultRefs, DefaultInputs]


def control_position_or_model(
    controller: CONTROLLER_T,
    model: DefaultTrainedModelType | None,
    qdes_func: RefFuncType,
    dqdes_func: RefFuncType,
    duration: float,
    logger: Logger | None = None,
    log_filename: str | Path | None = None,
    time_updater: str = "elapsed",
    header_text: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    reset_logger(logger, log_filename)
    comm, ctrl, state = controller
    ctrl_adapter = DefaultCtrlAdapter(ctrl, model)
    ca, cb = np.zeros(ctrl.dof, dtype=float), np.zeros(ctrl.dof, dtype=float)
    timer = Timer(rate=ctrl.freq)
    current_time = select_time_updater(timer, time_updater)

    timer.start()
    t = 0.0
    while t < duration:
        sys.stdout.write(f"\r{header_text} [{t:6.2f}/{duration:.2f}]")
        t = current_time()
        rq, rdq, rpa, rpb = state.get_raw_states()
        q, dq, pa, pb = state.get_states()
        qdes, dqdes = qdes_func(t), dqdes_func(t)
        ca, cb = ctrl_adapter.update(t, q, dq, pa, pb, qdes, dqdes)
        comm.send_commands(ca, cb)
        if logger is not None:
            logger.store(t, rq, rdq, rpa, rpb, q, dq, pa, pb, ca, cb, qdes, dqdes)
        timer.block()
    sys.stdout.write("\n")
    # Return the last commands that have been sent to the valve.
    return ca, cb


# Local Variables:
# jinx-local-words: "Params arg cb dataset dq dqdes maxabs minmax mlp noqa npqa params pb qdes regressor scaler"
# End:
