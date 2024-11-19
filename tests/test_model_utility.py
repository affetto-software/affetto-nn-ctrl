from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pyplotutil.datautil import Data
from sklearn import datasets
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor

from affetto_nn_ctrl import ROOT_DIR_PATH
from affetto_nn_ctrl.event_logging import event_logger, start_event_logging
from affetto_nn_ctrl.model_utility import (
    DataAdapterBase,
    DataAdapterParamsBase,
    DefaultInputs,
    DefaultRefs,
    DefaultStates,
    InputsBase,
    RefsBase,
    Regressor,
    StatesBase,
    train_model,
)

try:
    from . import TESTS_DATA_DIR_PATH, assert_file_contents
except ImportError:
    sys.path.append(str(ROOT_DIR_PATH))
    from tests import TESTS_DATA_DIR_PATH, assert_file_contents  # type: ignore[reportMissingImports]

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired

if TYPE_CHECKING:
    from pathlib import Path


def make_prediction_data_path(
    base_directory: Path,
    adapter_name: str,
    model_name: str,
    **model_params: str | float,
) -> Path:
    filename = f"{adapter_name}_{model_name}"
    if len(model_params):
        joined = "_".join("-".join(map(str, x)) for x in model_params.items())
        filename += f"_{joined}"
    filename += ".csv"
    return base_directory / filename


@dataclass
class SimpleDataAdapterParams(DataAdapterParamsBase):
    feature_index: list[int]
    target_index: list[int]


class SimpleStates(StatesBase):
    f0: np.ndarray
    f1: NotRequired[np.ndarray]
    t0: np.ndarray


class SimpleDataAdapter(DataAdapterBase[SimpleDataAdapterParams, SimpleStates, RefsBase, InputsBase]):
    def __init__(self, params: SimpleDataAdapterParams) -> None:
        super().__init__(params)

    def make_feature(self, dataset: Data) -> np.ndarray:
        feature_keys = [f"f{x}" for x in self.params.feature_index]
        feature_data = dataset.dataframe[feature_keys]
        return feature_data.to_numpy()

    def make_target(self, dataset: Data) -> np.ndarray:
        target_keys = [f"t{x}" for x in self.params.target_index]
        target_data = dataset.dataframe[target_keys]
        return target_data.to_numpy()

    def make_model_input(self, t: float, states: SimpleStates, refs: RefsBase) -> np.ndarray:
        _ = refs
        inputs: list[float] = []
        for i in self.params.feature_index:
            key = f"f{i}"
            values = states[key]  # type: ignore[literal-required]
            if isinstance(values, np.ndarray):
                inputs.extend(values)
            elif isinstance(values, Callable):  # type: ignore[arg-type]
                # https://github.com/python/mypy/issues/3060
                inputs.extend(values(t))
            else:
                msg = f"unsupported type: states[{key}] = {values} ({type(values)})"
                raise TypeError(msg)
        return np.asarray([inputs], dtype=float)

    def make_ctrl_input(self, y: np.ndarray, base_inputs: InputsBase) -> tuple[np.ndarray, ...]:
        _ = base_inputs
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = np.ravel(y)
        if not isinstance(y, Iterable):
            return (y,)
        return tuple(y)

    def reset(self) -> None:
        pass


class TestSimpleDataAdapter:
    def make_dataset(self, n_samples: int, n_features: int, n_targets: int) -> tuple[Data, Data]:
        X, y = datasets.make_regression(  # type: ignore[] # noqa: N806
            n_samples,
            n_features=n_features,
            n_informative=n_features,
            n_targets=n_targets,
            bias=100,
            noise=4.0,
            random_state=42,
        )
        if y.ndim == 1:
            y = np.atleast_2d(y).T
        data = np.hstack((X, y))
        columns = [f"f{x}" for x in range(n_features)] + [f"t{x}" for x in range(n_targets)]
        dataset = Data(pd.DataFrame(dict(zip(columns, data.T, strict=True))))
        return dataset.split_by_row(int(0.75 * 20))

    def predict(self, test_dataset: Data, adapter: SimpleDataAdapter, model: Regressor) -> np.ndarray:
        prediction: list[tuple[np.ndarray, ...]] = []
        keys = [f"f{x}" for x in adapter.params.feature_index] + [f"t{x}" for x in adapter.params.target_index]
        for x_input in test_dataset:
            kw = dict(zip(keys, map(np.atleast_1d, x_input), strict=True))
            t = 0
            x = adapter.make_model_input(t, kw, {})  # type: ignore[arg-type]
            y = model.predict(x)
            c = adapter.make_ctrl_input(np.asarray(y), {})
            prediction.append(c)
        return np.asarray(prediction)

    def generate_prediction_data(
        self,
        output: Path,
        train_dataset: Data,
        test_dataset: Data,
        adapter: SimpleDataAdapter,
        model: Regressor,
    ) -> np.ndarray:
        model = train_model(model, train_dataset, adapter)
        prediction = self.predict(test_dataset, adapter, model)
        np.savetxt(output, prediction)
        event_logger().info("Expected data for SimpleDataAdapter generated: %s", output)
        return prediction

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.parametrize(
        ("model", "kw", "name"),
        [
            (LinearRegression(), {}, "linear"),
            (Ridge(alpha=0.1), {"alpha": 0.1}, "ridge"),
            (Ridge(alpha=1.0), {"alpha": 1.0}, "ridge"),
            (MLPRegressor(random_state=42, max_iter=200), {"max_iter": 200}, "mlp"),
            (MLPRegressor(random_state=42, max_iter=500), {"max_iter": 500}, "mlp"),
            (MLPRegressor(random_state=42, max_iter=800), {"max_iter": 800}, "mlp"),
        ],
    )
    def test_simple_data_adapter(
        self,
        make_work_directory: Path,
        model: Regressor,
        kw: dict[str, str | float],
        name: str,
    ) -> None:
        train_dataset, test_dataset = self.make_dataset(n_samples=20, n_features=1, n_targets=1)
        adapter = SimpleDataAdapter(SimpleDataAdapterParams(feature_index=[0], target_index=[0]))
        output = make_prediction_data_path(make_work_directory, "simple", name, **kw)
        self.generate_prediction_data(output, train_dataset, test_dataset, adapter, model)
        assert_file_contents(TESTS_DATA_DIR_PATH / output.name, output)


def check_expected_data_for_simple_data_adapter(
    train_dataset: Data,
    test_dataset: Data,
    adapter: SimpleDataAdapter,
    prediction: np.ndarray,
    title: str | None,
) -> None:
    x_train = adapter.make_feature(train_dataset)
    y_train = adapter.make_target(train_dataset)
    x_test = adapter.make_feature(test_dataset)
    y_test = adapter.make_target(test_dataset)

    plt.plot(x_train[:, 0], y_train[:, 0], "k.", label="observed")
    plt.plot(x_test[:, 0], y_test[:, 0], "b.", label="test")
    plt.plot(x_test[:, 0], prediction[:, 0], "r.", label="prediction")
    if title is not None:
        plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    plt.close()


def generate_expected_data_for_simple_data_adapter(*, show_plot: bool = True) -> None:
    generator = TestSimpleDataAdapter()
    train_dataset, test_dataset = generator.make_dataset(n_samples=20, n_features=1, n_targets=1)
    adapter = SimpleDataAdapter(SimpleDataAdapterParams(feature_index=[0], target_index=[0]))
    TESTS_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)

    models: list[tuple[Regressor, dict[str, str | float], str]] = [
        (LinearRegression(), {}, "linear"),
        (Ridge(alpha=0.1), {"alpha": 0.1}, "ridge"),
        (Ridge(alpha=1.0), {"alpha": 1.0}, "ridge"),
        (MLPRegressor(random_state=42, max_iter=200), {"max_iter": 200}, "mlp"),
        (MLPRegressor(random_state=42, max_iter=500), {"max_iter": 500}, "mlp"),
        (MLPRegressor(random_state=42, max_iter=800), {"max_iter": 800}, "mlp"),
    ]
    for model, kw, name in models:
        output = make_prediction_data_path(TESTS_DATA_DIR_PATH, "simple", name, **kw)
        prediction = generator.generate_prediction_data(output, train_dataset, test_dataset, adapter, model)
        if show_plot:
            params = ",".join(":".join(map(str, x)) for x in kw.items())
            event_logger().info("Plotting expected data for %s (%s)", name, params)
            check_expected_data_for_simple_data_adapter(train_dataset, test_dataset, adapter, prediction, output.stem)


@dataclass
class JointDataAdapterParams(DataAdapterParamsBase):
    active_joints: list[int]


class JointDataAdapter(DataAdapterBase[JointDataAdapterParams, DefaultStates, DefaultRefs, DefaultInputs]):
    def __init__(self, params: JointDataAdapterParams) -> None:
        super().__init__(params)

    def get_keys(
        self,
        symbols: Iterable[str],
        joints: Iterable[int] | None = None,
        *,
        add_t: bool = False,
    ) -> list[str]:
        keys: list[str] = []
        if add_t:
            keys.append("t")
        if joints is None:
            joints = self.params.active_joints
        for s in symbols:
            keys.extend([f"{s}{i}" for i in joints])
        return keys

    def extract_data(
        self,
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

    def make_feature(self, dataset: Data) -> np.ndarray:
        states = self.extract_data(dataset, self.get_keys(["q", "dq", "pa", "pb"]), -1)
        reference = self.extract_data(dataset, self.get_keys(["q"]), 1, self.get_keys(["qdes"]))
        feature_data = pd.concat((states, reference), axis=1)
        return feature_data.to_numpy()

    def make_target(self, dataset: Data) -> np.ndarray:
        ctrl_input = self.extract_data(dataset, self.get_keys(["ca", "cb"]), 1)
        return ctrl_input.to_numpy()

    def make_model_input(self, t: float, states: DefaultStates, refs: DefaultRefs) -> np.ndarray:
        q = states["q"][self.params.active_joints]
        dq = states["dq"][self.params.active_joints]
        pa = states["pa"][self.params.active_joints]
        pb = states["pb"][self.params.active_joints]
        qdes = refs["qdes"](t)[self.params.active_joints]
        model_input = np.concatenate((q, dq, pa, pb, qdes))
        return np.atleast_2d(model_input)

    def make_ctrl_input(self, y: np.ndarray, base_inputs: DefaultInputs) -> tuple[np.ndarray, ...]:
        ca, cb = base_inputs["ca"], base_inputs["cb"]
        n = len(self.params.active_joints)
        ca[self.params.active_joints] = y[:n]
        cb[self.params.active_joints] = y[n:]
        return ca, cb

    def reset(self) -> None:
        pass


TOY_JOINT_DATA_TXT = """\
t,q0,q5,dq0,dq5,pa0,pa5,pb0,pb5,ca0,ca5,cb0,cb5
4.00,0.81,20.80,-2.96,-25.81,378.01,377.55,401.96,418.98,169.11,166.36,170.89,173.64
4.03,0.73,20.09,-1.60,-23.02,377.95,378.46,401.70,418.17,169.23,169.07,170.77,170.93
4.07,0.75,19.42,1.56,-21.88,377.68,379.30,401.75,417.06,169.17,172.22,170.83,167.78
4.10,0.83,18.70,1.28,-15.13,377.97,380.30,401.94,416.46,169.02,175.92,170.98,164.08
4.13,0.83,18.34,-0.41,-10.35,378.01,380.76,401.63,415.07,169.04,179.60,170.96,160.40
4.17,0.84,18.11,0.88,-6.34,377.76,381.88,401.53,412.69,169.01,183.58,170.99,156.42
4.20,0.86,17.99,-0.40,-1.26,377.97,383.14,402.25,409.73,168.98,187.50,171.02,152.50
4.23,0.81,17.99,-2.04,0.57,377.97,386.51,401.27,407.22,169.09,191.25,170.91,148.75
4.27,0.76,18.02,-1.38,0.76,378.28,395.33,400.99,403.54,169.18,194.99,170.82,145.01
4.30,0.75,18.07,0.56,1.78,378.36,408.76,401.51,394.00,169.17,198.67,170.83,141.33
4.33,0.76,18.14,-0.03,2.86,378.03,417.22,401.71,384.80,169.16,202.33,170.84,137.67
4.37,0.77,18.31,0.96,7.36,378.18,425.04,401.54,376.35,169.13,205.84,170.87,134.16
4.40,0.86,18.99,1.17,24.90,378.30,432.04,401.35,367.69,168.97,208.50,171.03,131.50
4.43,0.78,20.14,-3.89,45.59,378.29,434.53,400.87,362.90,169.17,210.27,170.83,129.73
4.47,0.71,22.01,-0.56,70.96,378.34,436.59,400.93,359.30,169.27,210.46,170.73,129.54
4.50,0.79,25.49,1.36,94.34,378.26,439.11,401.66,355.34,169.10,207.85,170.90,132.15
4.53,0.70,28.57,-4.70,106.93,378.18,440.78,401.46,353.43,169.32,205.37,170.68,134.63
4.57,0.63,32.02,0.39,118.53,378.19,442.97,400.78,352.27,169.41,201.87,170.59,138.13
4.60,0.81,36.97,4.11,125.99,378.48,445.54,400.83,351.05,169.04,195.71,170.96,144.29
4.63,0.84,40.85,-0.24,131.02,378.16,446.15,401.54,351.01,169.02,190.63,170.98,149.37
"""


@pytest.fixture(scope="session")
def toy_joint_data() -> Data:
    return Data(TOY_JOINT_DATA_TXT)


def main() -> None:
    import sys

    start_event_logging(sys.argv, logging_level="INFO")
    msg = f"""\
    Provide a number to generate expected data.

    MENU:
      1) Expected data for testing SimpleDataAdapter class
      2) Expected data for testing JointDataAdapter class
      3) Expected data for testing CtrlAdapter class

    EXAMPLE:
      $ python {' '.join(sys.argv)} 1
"""
    if len(sys.argv) > 1:
        match sys.argv[1]:
            case "1":
                generate_expected_data_for_simple_data_adapter(show_plot=True)
            case "2":
                pass
            case _:
                raise RuntimeError(msg)
    else:
        raise RuntimeError(msg)


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "Ctrl arg cb csv dq iter mlp noqa params pb qdes sklearn"
# End:
