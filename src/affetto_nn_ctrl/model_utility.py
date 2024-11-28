from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Protocol, TypeAlias, TypedDict, TypeVar, cast

import joblib
import numpy as np
import pandas as pd
from affctrllib import Logger, Timer
from pyplotutil.datautil import Data
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

from affetto_nn_ctrl import RefFuncType
from affetto_nn_ctrl.control_utility import reset_logger, select_time_updater
from affetto_nn_ctrl.event_logging import event_logger

if TYPE_CHECKING:
    from affctrllib import AffPosCtrl

    from affetto_nn_ctrl import CONTROLLER_T
    from affetto_nn_ctrl._typing import T, Unknown

if sys.version_info >= (3, 11):
    from typing import NotRequired

    import tomllib
else:
    import tomli as tomllib
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
    qdes: RefFuncType
    dqdes: RefFuncType


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

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"

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


def _get_keys(symbols: Iterable[str], active_joints: list[int], *, add_t: bool = False) -> list[str]:
    keys: list[str] = []
    if add_t:
        keys.append("t")
    for s in symbols:
        keys.extend([f"{s}{i}" for i in active_joints])
    return keys


@dataclass
class PreviewRefParams(DataAdapterParamsBase):
    active_joints: list[int]
    dt: float
    preview_step: int
    include_dqdes: bool = False


class PreviewRef(DataAdapterBase[PreviewRefParams, DefaultStates, DefaultRefs, DefaultInputs]):
    def __init__(self, params: PreviewRefParams) -> None:
        super().__init__(params)
        if params.preview_step < 1:
            msg = f"PreviewRefParams.preview_step must be larger than or equal to 1: {params.preview_step}"
            raise ValueError(msg)

    def make_feature(self, dataset: Data) -> np.ndarray:
        joints = self.params.active_joints
        shift = self.params.preview_step
        states = extract_data(dataset, _get_keys(["q", "dq", "pa", "pb"], joints), end=-shift)
        ref_keys: list[str] = ["q"]
        if self.params.include_dqdes:
            ref_keys.append("dq")
        reference = extract_data(
            dataset,
            _get_keys(ref_keys, joints),
            start=shift,
            keys_replace=_get_keys([f"{x}des" for x in ref_keys], joints),
        )
        feature_data = pd.concat((states, reference), axis=1)
        return feature_data.to_numpy()

    def make_target(self, dataset: Data) -> np.ndarray:
        joints = self.params.active_joints
        shift = self.params.preview_step
        ctrl_input = extract_data(dataset, _get_keys(["ca", "cb"], joints), end=-shift)
        return ctrl_input.to_numpy()

    def make_model_input(self, t: float, states: DefaultStates, refs: DefaultRefs) -> np.ndarray:
        joints = self.params.active_joints
        preview_time = self.params.preview_step * self.params.dt
        q = states["q"][joints]
        dq = states["dq"][joints]
        pa = states["pa"][joints]
        pb = states["pb"][joints]
        qdes = refs["qdes"](t + preview_time)[joints]
        if self.params.include_dqdes:
            dqdes = refs["dqdes"](t + preview_time)[joints]
            model_input = np.concatenate((q, dq, pa, pb, qdes, dqdes))
        else:
            model_input = np.concatenate((q, dq, pa, pb, qdes))  # concatenate feature vectors horizontally
        # Make matrix that has 1 row and n_features columns.
        return np.atleast_2d(model_input)

    def make_ctrl_input(self, y: np.ndarray, base_inputs: DefaultInputs) -> tuple[np.ndarray, ...]:
        # `y` has has 1 row and n_targets columns.
        joints = self.params.active_joints
        y = np.ravel(y)
        ca, cb = base_inputs["ca"], base_inputs["cb"]
        n = len(joints)
        ca[joints] = y[:n]
        cb[joints] = y[n:]
        return ca, cb

    def reset(self) -> None:
        pass


@dataclass
class DelayStatesParams(DataAdapterParamsBase):
    active_joints: list[int]
    dt: float
    ctrl_step: int
    delay_step: int
    include_dqdes: bool = False


class DelayStates(DataAdapterBase[DelayStatesParams, DefaultStates, DefaultRefs, DefaultInputs]):
    def __init__(self, params: DelayStatesParams) -> None:
        super().__init__(params)
        if params.ctrl_step < 1:
            msg = f"DelayStatesParams.ctrl_step must be larger than or equal to 1: {params.ctrl_step}"
            raise ValueError(msg)
        if params.delay_step < 0:
            msg = f"DelayStatesParams.delay_step must be larger than or equal to 0: {params.delay_step}"
            raise ValueError(msg)
        self.reset()

    def make_feature(self, dataset: Data) -> np.ndarray:
        joints = self.params.active_joints
        ctrl_step = self.params.ctrl_step
        delay_step = self.params.delay_step
        shift = ctrl_step + delay_step

        states_keys = _get_keys(["q", "dq", "pa", "pb"], joints)
        refs: list[str] = ["q"]
        if self.params.include_dqdes:
            refs.append("dq")
        ref_keys = _get_keys(refs, joints)

        delayed_states = extract_data(dataset, states_keys, end=-shift)
        current_states = extract_data(dataset, states_keys, start=delay_step, end=-ctrl_step)
        reference = extract_data(
            dataset,
            ref_keys,
            start=shift,
            keys_replace=_get_keys([f"{x}des" for x in refs], joints),
        )
        feature_data = pd.concat((delayed_states, current_states, reference), axis=1)
        return feature_data.to_numpy()

    def make_target(self, dataset: Data) -> np.ndarray:
        joints = self.params.active_joints
        ctrl_step = self.params.ctrl_step
        delay_step = self.params.delay_step
        ctrl_input = extract_data(dataset, _get_keys(["ca", "cb"], joints), start=delay_step, end=-ctrl_step)
        return ctrl_input.to_numpy()

    def make_model_input(self, t: float, states: DefaultStates, refs: DefaultRefs) -> np.ndarray:
        joints = self.params.active_joints
        q = states["q"][joints]
        dq = states["dq"][joints]
        pa = states["pa"][joints]
        pb = states["pb"][joints]
        current_states = np.concatenate((q, dq, pa, pb))
        self.states_queue.append(current_states)
        delayed_states = self.states_queue.popleft()

        qdes = refs["qdes"](t)[joints]
        if self.params.include_dqdes:
            dqdes = refs["dqdes"](t)[joints]
            model_input = np.concatenate((delayed_states, current_states, qdes, dqdes))
        else:
            model_input = np.concatenate((delayed_states, current_states, qdes))
        return np.atleast_2d(model_input)

    def make_ctrl_input(self, y: np.ndarray, base_inputs: DefaultInputs) -> tuple[np.ndarray, ...]:
        # `y` has has 1 row and n_targets columns.
        joints = self.params.active_joints
        y = np.ravel(y)
        ca, cb = base_inputs["ca"], base_inputs["cb"]
        n = len(joints)
        ca[joints] = y[:n]
        cb[joints] = y[n:]
        return ca, cb

    def reset(self) -> None:
        n_states = len(self.params.active_joints) * len(["q", "dq", "pa", "pb"])
        self.states_queue = deque([np.zeros(shape=n_states, dtype=float)] * self.params.delay_step)


@dataclass
class DelayStatesAllParams(DataAdapterParamsBase):
    active_joints: list[int]
    dt: float
    ctrl_step: int
    delay_step: int
    include_dqdes: bool = False


class DelayStatesAll(DataAdapterBase[DelayStatesAllParams, DefaultStates, DefaultRefs, DefaultInputs]):
    def __init__(self, params: DelayStatesAllParams) -> None:
        super().__init__(params)
        if params.ctrl_step < 1:
            msg = f"DelayStatesParams.ctrl_step must be larger than or equal to 1: {params.ctrl_step}"
            raise ValueError(msg)
        if params.delay_step < 0:
            msg = f"DelayStatesParams.delay_step must be larger than or equal to 0: {params.delay_step}"
            raise ValueError(msg)
        self.reset()

    def make_feature(self, dataset: Data) -> np.ndarray:
        joints = self.params.active_joints
        ctrl_step = self.params.ctrl_step
        delay_step = self.params.delay_step
        shift = ctrl_step + delay_step

        states_keys = _get_keys(["q", "dq", "pa", "pb"], joints)
        refs: list[str] = ["q"]
        if self.params.include_dqdes:
            refs.append("dq")
        ref_keys = _get_keys(refs, joints)

        states = extract_data(dataset, states_keys, start=delay_step, end=-ctrl_step)
        for i in range(1, delay_step + 1):
            start = delay_step - i
            end = ctrl_step + i
            delayed_states = extract_data(dataset, states_keys, start=start, end=-end)
            states = pd.concat((delayed_states, states), axis=1)
        reference = extract_data(
            dataset,
            ref_keys,
            start=shift,
            keys_replace=_get_keys([f"{x}des" for x in refs], joints),
        )
        feature_data = pd.concat((states, reference), axis=1)
        return feature_data.to_numpy()

    def make_target(self, dataset: Data) -> np.ndarray:
        joints = self.params.active_joints
        ctrl_step = self.params.ctrl_step
        delay_step = self.params.delay_step
        ctrl_input = extract_data(dataset, _get_keys(["ca", "cb"], joints), start=delay_step, end=-ctrl_step)
        return ctrl_input.to_numpy()

    def make_model_input(self, t: float, states: DefaultStates, refs: DefaultRefs) -> np.ndarray:
        joints = self.params.active_joints
        q = states["q"][joints]
        dq = states["dq"][joints]
        pa = states["pa"][joints]
        pb = states["pb"][joints]
        current_states = np.concatenate((q, dq, pa, pb))
        self.states_queue.append(current_states)
        concat_states = np.ravel(self.states_queue)
        self.states_queue.pop(0)

        qdes = refs["qdes"](t)[joints]
        if self.params.include_dqdes:
            dqdes = refs["dqdes"](t)[joints]
            model_input = np.concatenate((concat_states, qdes, dqdes))
        else:
            model_input = np.concatenate((concat_states, qdes))
        return np.atleast_2d(model_input)

    def make_ctrl_input(self, y: np.ndarray, base_inputs: DefaultInputs) -> tuple[np.ndarray, ...]:
        # `y` has has 1 row and n_targets columns.
        joints = self.params.active_joints
        y = np.ravel(y)
        ca, cb = base_inputs["ca"], base_inputs["cb"]
        n = len(joints)
        ca[joints] = y[:n]
        cb[joints] = y[n:]
        return ca, cb

    def reset(self) -> None:
        n_states = len(self.params.active_joints) * len(["q", "dq", "pa", "pb"])
        self.states_queue = [np.zeros(shape=n_states, dtype=float)] * self.params.delay_step


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


def load_model_config_file(config_file: str | Path) -> dict[str, Unknown]:
    with Path(config_file).open("rb") as f:
        return tomllib.load(f)


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


def _tweak_config_values(config: dict[str, Unknown]) -> dict[str, Unknown]:
    for key, value in config.items():
        if isinstance(value, str) and value.lower() == "none":
            config[key] = None

        match key:
            case "feature_range":
                config[key] = tuple(value)  # MinMaxScaler
            case "quantile_range":
                config[key] = tuple(value)  # RobustScaler
            case "hidden_layer_sizes":
                config[key] = tuple(value)  # MLPRegressor
    return config


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

    _tweak_config_values(config)
    if params_type is None:
        return _type(**config)
    return _type(params_type(**config))  # type: ignore[call-arg]


def update_config_by_selector(
    config: dict[str, Unknown],
    selector: str,
) -> dict[str, Unknown]:
    splitted = selector.split(".")
    config.update(name=splitted[0])
    if len(splitted) > 1:
        config.update(params=splitted[1])
    return config


def load_data_adapter(
    config: dict[str, Unknown],
    active_joints: list[int] | None = None,
    selector: str | None = None,
) -> DataAdapterBase:
    if selector is not None:
        config = update_config_by_selector(config, selector)
    adapter = _load_from_map(config, DATA_ADAPTER_MAP, "data adapter")
    if active_joints is not None:
        adapter.params.active_joints = active_joints
    event_logger().debug("Loaded data adapter: %s", adapter)
    return adapter


def load_scaler(config: dict[str, Unknown], selector: str | None = None) -> Scaler | None:
    if selector is not None:
        config = update_config_by_selector(config, selector)
    scaler = None
    if config.get("name", "x").lower() != "none":
        scaler = _load_from_map(config, SCALER_MAP, "scaler")
    event_logger().debug("Loaded scaler: %s", scaler)
    return scaler


def load_regressor(config: dict[str, Unknown], selector: str | None = None) -> Regressor:
    if selector is not None:
        config = update_config_by_selector(config, selector)
    regressor = _load_from_map(config, REGRESSOR_MAP, "regressor")
    event_logger().debug("Loaded regressor: %s", regressor)
    return regressor


def load_model(
    config: dict[str, Unknown],
    scaler_selector: str | None = None,
    regressor_selector: str | None = None,
) -> Regressor | Pipeline:
    scaler = load_scaler(config["scaler"], scaler_selector)
    regressor = load_regressor(config["regressor"], regressor_selector)
    if scaler is None:
        return regressor
    return make_pipeline(scaler, regressor)


def extract_data(
    dataset: Data,
    keys: Iterable[str],
    *,
    start: int | None = None,
    end: int | None = None,
    keys_replace: Iterable[str] | None = None,
) -> pd.DataFrame:
    subset = dataset.df.loc[:, keys]
    if start is None:
        start = 0
    if end is None:
        end = len(subset)
    extracted = subset[start:end]
    if keys_replace is not None:
        extracted = extracted.rename(columns=dict(zip(keys, keys_replace, strict=True)))
    return extracted.reset_index(drop=True)


def _load_datasets(
    data_file_or_directory: str | Path,
    glob_pattern: str,
    n_pickup: int | None,
) -> list[Data]:
    collection: list[Data] = []
    path = Path(data_file_or_directory)
    if path.is_dir():
        collection.extend(Data(x) for x in sorted(path.glob(glob_pattern)))
        if n_pickup is not None:
            collection = collection[:n_pickup]
        event_logger().debug("%s datasets loaded from %s", len(collection), path)
    else:
        collection.append(Data(path))
        event_logger().debug("dataset loaded from %s", path)

    return collection


def load_datasets(
    dataset_paths: str | Path | Iterable[str | Path],
    glob_pattern: str = "**/*.csv",
    n_pickup: int | None = None,
) -> list[Data]:
    if isinstance(dataset_paths, str | Path):
        dataset_paths = [dataset_paths]

    datasets: list[Data] = []
    for dataset_path in dataset_paths:
        datasets.extend(_load_datasets(dataset_path, glob_pattern, n_pickup))

    if len(datasets) == 0:
        msg = f"No dataset found with {glob_pattern}: {dataset_paths}"
        raise RuntimeError(msg)
    return datasets


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


def train_model(
    model: Regressor | Pipeline,
    datasets: Data | Iterable[Data],
    adapter: DataAdapterBase[DataAdapterParamsType, StatesType, RefsType, InputsType],
) -> TrainedModel:
    x_train, y_train = load_train_datasets(datasets, adapter)
    event_logger().debug("x_train.shape = %s", x_train.shape)
    event_logger().debug("y_train.shape = %s", y_train.shape)
    model = model.fit(x_train, y_train)
    return TrainedModel(model, adapter)


def dump_trained_model(trained_model: TrainedModel, output: str | Path) -> Path:
    joblib.dump(trained_model, output)
    return Path(output)


def load_trained_model(model_filepath: str | Path) -> TrainedModel:
    return joblib.load(model_filepath)


CtrlAdapterUpdater: TypeAlias = Callable[
    [float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, RefFuncType, RefFuncType],
    tuple[np.ndarray, np.ndarray],
]


class CtrlAdapter(Generic[DataAdapterParamsType, StatesType, RefsType, InputsType]):
    ctrl: AffPosCtrl
    model: TrainedModel[DataAdapterParamsType, StatesType, RefsType, InputsType]
    _updater: CtrlAdapterUpdater
    warmup_steps: int
    _n_steps: int

    def __init__(
        self,
        ctrl: AffPosCtrl,
        model: TrainedModel[DataAdapterParamsType, StatesType, RefsType, InputsType] | None,
        warmup_steps: int,
    ) -> None:
        self.ctrl = ctrl
        if model is None:
            self._updater = self.update_ctrl
        else:
            self.model = model
            self._updater = self.update_model
        self.warmup_steps = warmup_steps
        self._n_steps = 0

    def update_ctrl(
        self,
        t: float,
        q: np.ndarray,
        dq: np.ndarray,
        pa: np.ndarray,
        pb: np.ndarray,
        qdes: RefFuncType,
        dqdes: RefFuncType,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.ctrl.update(t, q, dq, pa, pb, qdes(t), dqdes(t))

    def update_model(
        self,
        t: float,
        q: np.ndarray,
        dq: np.ndarray,
        pa: np.ndarray,
        pb: np.ndarray,
        qdes: RefFuncType,
        dqdes: RefFuncType,
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
        qdes: RefFuncType,
        dqdes: RefFuncType,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._n_steps < self.warmup_steps:
            ca, cb = self.ctrl.update(t, q, dq, pa, pb, qdes(t), dqdes(t))
        else:
            ca, cb = self._updater(t, q, dq, pa, pb, qdes, dqdes)
        self._n_steps += 1
        return ca, cb


class DefaultCtrlAdapter(CtrlAdapter[DataAdapterParamsType, DefaultStates, DefaultRefs, DefaultInputs]):
    def __init__(
        self,
        ctrl: AffPosCtrl,
        model: TrainedModel[DataAdapterParamsType, DefaultStates, DefaultRefs, DefaultInputs] | None,
        warmup_steps: int,
    ) -> None:
        super().__init__(ctrl, model, warmup_steps)

    def update_model(
        self,
        t: float,
        q: np.ndarray,
        dq: np.ndarray,
        pa: np.ndarray,
        pb: np.ndarray,
        qdes: RefFuncType,
        dqdes: RefFuncType,
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
    time_updater: str = "accumulated",
    header_text: str = "",
    warmup_steps: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    reset_logger(logger, log_filename)
    comm, ctrl, state = controller
    ctrl_adapter = DefaultCtrlAdapter(ctrl, model, warmup_steps)
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
        ca, cb = ctrl_adapter.update(t, q, dq, pa, pb, qdes_func, dqdes_func)
        comm.send_commands(ca, cb)
        if logger is not None:
            logger.store(t, rq, rdq, rpa, rpb, q, dq, pa, pb, ca, cb, qdes_func(t), dqdes_func(t))
        timer.block()
    sys.stdout.write("\n")
    # Return the last commands that have been sent to the valve.
    return ca, cb


# Local Variables:
# jinx-local-words: "MLPRegressor Params apdater arg cb csv ctrl dataset datasets des dq dqdes maxabs minmax mlp noqa npqa params pb qdes quantile rb regressor scaler" # noqa: E501
# End:
