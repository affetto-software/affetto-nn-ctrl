from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt
import pandas as pd
import pandas.testing as pt
import pytest
from pyplotutil.datautil import Data
from sklearn import datasets
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

from affetto_nn_ctrl import ROOT_DIR_PATH
from affetto_nn_ctrl.event_logging import event_logger, start_event_logging
from affetto_nn_ctrl.model_utility import (
    DataAdapterBase,
    DataAdapterParamsBase,
    DefaultInputs,
    DefaultRefs,
    DefaultStates,
    DelayStates,
    DelayStatesAll,
    DelayStatesAllParams,
    DelayStatesParams,
    InputsBase,
    PreviewRef,
    PreviewRefParams,
    RefsBase,
    Regressor,
    Scaler,
    StatesBase,
    extract_data,
    load_data_adapter,
    load_regressor,
    load_scaler,
    load_train_datasets,
    train_model,
    update_config_by_selector,
)

try:
    from . import TESTS_DATA_DIR_PATH, assert_file_contents
except ImportError:
    sys.path.append(str(ROOT_DIR_PATH))
    from tests import TESTS_DATA_DIR_PATH, assert_file_contents  # type: ignore[reportMissingImports]

if sys.version_info >= (3, 11):
    from typing import NotRequired

    import tomllib
else:
    import tomli as tomllib
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

    @pytest.mark.slow
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


TOY_DATA_TXT = """\
a,b,c,d,e
1,2,3,4,5
6,7,8,9,10
11,12,13,14,15
16,17,18,19,20
21,22,23,24,25
"""


@pytest.fixture(scope="session")
def toy_data() -> Data:
    return Data(StringIO(TOY_DATA_TXT))


@pytest.mark.parametrize(
    ("keys", "shift", "expected"),
    [
        (["a", "c"], 0, "a,c\n1,3\n6,8\n11,13\n16,18\n21,23\n"),
        (("d",), 0, "d\n4\n9\n14\n19\n24\n"),
        (("b", "c"), 1, "b,c\n7,8\n12,13\n17,18\n22,23\n"),
        (["d", "e"], -1, "d,e\n4,5\n9,10\n14,15\n19,20\n"),
        (["d"], 2, "d\n14\n19\n24\n"),
        (["b"], -2, "b\n2\n7\n12\n"),
    ],
)
def test_extract_data(toy_data: Data, keys: Iterable[str], shift: int, expected: str) -> None:
    extracted_data = extract_data(toy_data, keys, shift)
    expected_data = pd.read_csv(StringIO(expected))
    n = len(expected_data)
    pt.assert_index_equal(extracted_data.index, pd.RangeIndex(n))
    pt.assert_frame_equal(extracted_data, expected_data)


@pytest.mark.parametrize(
    ("keys", "shift", "keys_replace", "expected"),
    [
        (["a", "c"], 0, ("A", "C"), "A,C\n1,3\n6,8\n11,13\n16,18\n21,23\n"),
        (("d",), 0, ["D"], "D\n4\n9\n14\n19\n24\n"),
        (("b", "c"), 1, ("boo", "ciao"), "boo,ciao\n7,8\n12,13\n17,18\n22,23\n"),
        (["d", "e"], -1, ["done", "echo"], "done,echo\n4,5\n9,10\n14,15\n19,20\n"),
    ],
)
def test_extract_data_replace_keys(
    toy_data: Data,
    keys: Iterable[str],
    keys_replace: Iterable[str],
    shift: int,
    expected: str,
) -> None:
    extracted_data = extract_data(toy_data, keys, shift, keys_replace)
    expected_data = pd.read_csv(StringIO(expected))
    n = len(expected_data)
    pt.assert_index_equal(extracted_data.index, pd.RangeIndex(n))
    pt.assert_frame_equal(extracted_data, expected_data)


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

    def make_feature(self, dataset: Data) -> np.ndarray:
        states = extract_data(dataset, self.get_keys(["q", "dq", "pa", "pb"]), -1)
        reference = extract_data(dataset, self.get_keys(["q"]), 1, self.get_keys(["qdes"]))
        feature_data = pd.concat((states, reference), axis=1)
        return feature_data.to_numpy()

    def make_target(self, dataset: Data) -> np.ndarray:
        ctrl_input = extract_data(dataset, self.get_keys(["ca", "cb"]), 1)
        return ctrl_input.to_numpy()

    def make_model_input(self, t: float, states: DefaultStates, refs: DefaultRefs) -> np.ndarray:
        _ = t
        q = states["q"][self.params.active_joints]
        dq = states["dq"][self.params.active_joints]
        pa = states["pa"][self.params.active_joints]
        pb = states["pb"][self.params.active_joints]
        qdes = refs["qdes"][self.params.active_joints]
        model_input = np.concatenate((q, dq, pa, pb, qdes))  # concatenate feature vectors horizontally
        # Make matrix that has 1 row and n_features columns.
        return np.atleast_2d(model_input)

    def make_ctrl_input(self, y: np.ndarray, base_inputs: DefaultInputs) -> tuple[np.ndarray, ...]:
        # `y` has has 1 row and n_targets columns.
        y = np.ravel(y)
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
    return Data(StringIO(TOY_JOINT_DATA_TXT))


class TestJointDataAdapter:
    @pytest.fixture
    def adapter(self) -> JointDataAdapter:
        return JointDataAdapter(JointDataAdapterParams(active_joints=[5]))

    def test_make_feature(self, toy_joint_data: Data, adapter: JointDataAdapter) -> None:
        expected_data_text = """\
20.80,-25.81,377.55,418.98,20.09
20.09,-23.02,378.46,418.17,19.42
19.42,-21.88,379.30,417.06,18.70
18.70,-15.13,380.30,416.46,18.34
18.34,-10.35,380.76,415.07,18.11
18.11, -6.34,381.88,412.69,17.99
17.99, -1.26,383.14,409.73,17.99
17.99,  0.57,386.51,407.22,18.02
18.02,  0.76,395.33,403.54,18.07
18.07,  1.78,408.76,394.00,18.14
18.14,  2.86,417.22,384.80,18.31
18.31,  7.36,425.04,376.35,18.99
18.99, 24.90,432.04,367.69,20.14
20.14, 45.59,434.53,362.90,22.01
22.01, 70.96,436.59,359.30,25.49
25.49, 94.34,439.11,355.34,28.57
28.57,106.93,440.78,353.43,32.02
32.02,118.53,442.97,352.27,36.97
36.97,125.99,445.54,351.05,40.85
"""
        feature = adapter.make_feature(toy_joint_data)
        expected = np.loadtxt(StringIO(expected_data_text), delimiter=",")
        nt.assert_array_equal(feature, expected)

    def test_make_target(self, toy_joint_data: Data, adapter: JointDataAdapter) -> None:
        expected_data_text = """\
169.07,170.93
172.22,167.78
175.92,164.08
179.60,160.40
183.58,156.42
187.50,152.50
191.25,148.75
194.99,145.01
198.67,141.33
202.33,137.67
205.84,134.16
208.50,131.50
210.27,129.73
210.46,129.54
207.85,132.15
205.37,134.63
201.87,138.13
195.71,144.29
190.63,149.37
"""
        target = adapter.make_target(toy_joint_data)
        expected = np.loadtxt(StringIO(expected_data_text), delimiter=",")
        nt.assert_array_equal(target, expected)

    def test_make_model_input(self, adapter: JointDataAdapter) -> None:
        dof = 6
        q = np.arange(dof, dtype=float)
        dq = np.arange(dof, dtype=float) * 0.01
        pa = np.arange(dof, dtype=float) + 300
        pb = np.arange(dof, dtype=float) + 10
        qdes = q + 4

        t = 4.0
        x = adapter.make_model_input(t, {"q": q, "dq": dq, "pa": pa, "pb": pb}, {"qdes": qdes})
        expected = np.array([[5, 0.05, 305, 15, 9]], dtype=float)
        nt.assert_array_equal(x, expected)

    def test_make_ctrl_input(self, adapter: JointDataAdapter) -> None:
        dof = 6
        base_ca = np.arange(dof, dtype=float) + 200
        base_cb = np.arange(dof, dtype=float) + 20
        y = np.array([[231, 28]], dtype=float)
        ca, cb = adapter.make_ctrl_input(y, {"ca": base_ca, "cb": base_cb})
        expected_ca = np.array([200, 201, 202, 203, 204, 231], dtype=float)
        expected_cb = np.array([20, 21, 22, 23, 24, 28], dtype=float)
        nt.assert_array_equal(ca, expected_ca)
        nt.assert_array_equal(cb, expected_cb)

    def predict(self, test_dataset: Data, adapter: JointDataAdapter, model: Regressor) -> tuple[np.ndarray, float]:
        x_test, y_true = load_train_datasets(test_dataset, adapter)
        y_pred = model.predict(x_test)
        score = model.score(x_test, y_true)
        return y_pred, score

    def generate_prediction_data(
        self,
        output: Path,
        train_datasets: Iterable[Data],
        test_dataset: Data,
        adapter: JointDataAdapter,
        model: Regressor,
    ) -> tuple[np.ndarray, float]:
        model = train_model(model, train_datasets, adapter)
        y_pred, score = self.predict(test_dataset, adapter, model)
        np.savetxt(output, y_pred)
        event_logger().info("Expected data for JointDataAdapter generated: %s", output)
        event_logger().info("Score: %s", score)
        return y_pred, score

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.parametrize(
        ("model", "kw", "name", "expected_score"),
        [
            (LinearRegression(), {}, "linear", 0.43617892031416844),
            (Ridge(alpha=0.1), {"alpha": 0.1}, "ridge", 0.436189557255996),
            (Ridge(alpha=1.0), {"alpha": 1.0}, "ridge", 0.4362793601129537),
            (MLPRegressor(random_state=42, max_iter=200), {"max_iter": 200}, "mlp", 0.46797910150668864),
            (
                MLPRegressor(random_state=42, hidden_layer_sizes=(100, 100), max_iter=500),
                {"hidden_layer_sizes": "100-100", "max_iter": 500},
                "mlp",
                0.5007867097957437,
            ),
            (
                MLPRegressor(random_state=42, hidden_layer_sizes=(200,), max_iter=500),
                {"hidden_layer_sizes": 200, "max_iter": 500},
                "mlp",
                0.4847716669330141,
            ),
        ],
    )
    def test_model(
        self,
        make_work_directory: Path,
        model: Regressor,
        kw: dict[str, str | float],
        name: str,
        expected_score: float,
    ) -> None:
        train_datasets = map(Data, TESTS_DATA_DIR_PATH.glob("motion_data_00*.csv"))
        test_dataset = Data(TESTS_DATA_DIR_PATH / "motion_data_010.csv")
        adapter = JointDataAdapter(JointDataAdapterParams(active_joints=[5]))
        output = make_prediction_data_path(make_work_directory, "joint", name, **kw)
        y_test, score = self.generate_prediction_data(output, train_datasets, test_dataset, adapter, model)
        assert_file_contents(TESTS_DATA_DIR_PATH / output.name, output)
        assert expected_score == pytest.approx(score)


def check_expected_data_for_joint_data_adapter(
    test_dataset: Data,
    adapter: JointDataAdapter,
    y_pred: np.ndarray,
    title: str | None,
) -> None:
    x_test, y_true = load_train_datasets(test_dataset, adapter)

    (line,) = plt.plot(y_true[:, 0], ls="--", label="ca (true)")
    plt.plot(y_pred[:, 0], c=line.get_color(), label="ca (pred)")

    (line,) = plt.plot(y_true[:, 1], ls="--", label="cb (true)")
    plt.plot(y_pred[:, 1], c=line.get_color(), label="cb (pred)")

    if title is not None:
        plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    plt.close()


def generate_expected_data_for_joint_data_adapter(*, show_plot: bool = True) -> None:
    generator = TestJointDataAdapter()
    adapter = JointDataAdapter(JointDataAdapterParams(active_joints=[5]))
    TESTS_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
    train_datasets = list(map(Data, TESTS_DATA_DIR_PATH.glob("motion_data_00*.csv")))
    test_dataset = Data(TESTS_DATA_DIR_PATH / "motion_data_010.csv")

    models: list[tuple[Regressor, dict[str, str | float], str]] = [
        (LinearRegression(), {}, "linear"),
        (Ridge(alpha=0.1), {"alpha": 0.1}, "ridge"),
        (Ridge(alpha=1.0), {"alpha": 1.0}, "ridge"),
        (MLPRegressor(random_state=42, max_iter=200), {"max_iter": 200}, "mlp"),
        (
            MLPRegressor(random_state=42, hidden_layer_sizes=(100, 100), max_iter=500),
            {"hidden_layer_sizes": "100-100", "max_iter": 500},
            "mlp",
        ),
        (
            MLPRegressor(random_state=42, hidden_layer_sizes=(200,), max_iter=500),
            {"hidden_layer_sizes": 200, "max_iter": 500},
            "mlp",
        ),
    ]
    for model, kw, name in models:
        output = make_prediction_data_path(TESTS_DATA_DIR_PATH, "joint", name, **kw)
        y_pred, score = generator.generate_prediction_data(output, train_datasets, test_dataset, adapter, model)
        if show_plot:
            params = ",".join(":".join(map(str, x)) for x in kw.items())
            title = f"{name} ({params}), score: {score}"
            event_logger().info("Plotting expected data for %s (%s)", name, params)
            check_expected_data_for_joint_data_adapter(test_dataset, adapter, y_pred, title)


DATA_ADAPTER_CONFIG_TEXT = """\
[model.adapter]
name = "preview-ref"
params = "default"
active_joints = [2, 3, 4, 5]

[model.adapter.preview-ref.default]
ctrl_step = 1
preview_step = 5

[model.adapter.delay-states.default]
ctrl_step = 1
delay_step = 7

[model.adapter.delay-states-all.default]
active_joints = [2, 5]
ctrl_step = 1
delay_step = 8

[model.adapter.delay-states-all.non-default]
ctrl_step = 2
delay_step = 4
"""


@pytest.fixture
def data_adapter_config() -> dict:
    return tomllib.loads(DATA_ADAPTER_CONFIG_TEXT)


def test_load_data_adapter_typical_config(data_adapter_config: dict) -> None:
    config = data_adapter_config["model"]["adapter"]
    actual = load_data_adapter(config)
    expected = PreviewRef(PreviewRefParams([2, 3, 4, 5], ctrl_step=1, preview_step=5))
    assert type(actual) is type(expected)
    assert actual.params == expected.params


@pytest.mark.parametrize(
    ("selector", "active_joints", "expected"),
    [
        ("delay-states", [2, 3], DelayStates(DelayStatesParams([2, 3], ctrl_step=1, delay_step=7))),
        (
            "delay-states-all.non-default",
            None,
            DelayStatesAll(DelayStatesAllParams([2, 3, 4, 5], ctrl_step=2, delay_step=4)),
        ),
        (
            "delay-states-all.default",
            [1, 2, 3],
            DelayStatesAll(DelayStatesAllParams([1, 2, 3], ctrl_step=1, delay_step=8)),
        ),
    ],
)
def test_load_data_adapter_config_modified_by_user(
    data_adapter_config: dict,
    selector: str,
    active_joints: list[int] | None,
    expected: DataAdapterBase,
) -> None:
    config = data_adapter_config["model"]["adapter"]
    config = update_config_by_selector(config, selector)
    actual = load_data_adapter(config, active_joints)
    assert type(actual) is type(expected)
    assert actual.params == expected.params


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"name": "preview-ref", "active_joints": [2, 3]}, PreviewRef(PreviewRefParams([2, 3]))),
        (
            {"name": "delay-states", "active_joints": [0, 1, 2], "ctrl_step": 2},
            DelayStates(DelayStatesParams([0, 1, 2], ctrl_step=2)),
        ),
        (
            {"name": "delay-states-all", "active_joints": [5], "delay_step": 5},
            DelayStatesAll(DelayStatesAllParams([5], delay_step=5)),
        ),
        (
            {
                "name": "preview-ref",
                "params": "default",
                "active_joints": [1, 3],
                "preview-ref": {"default": {"ctrl_step": 3}},
            },
            PreviewRef(PreviewRefParams([1, 3], ctrl_step=3)),
        ),
        (
            {
                "name": "delay-states",
                "params": "good-params",
                "active_joints": [2, 3, 4],
                "preview-ref": {"default": {"ctrl_step": 2}},
                "delay-states": {"default": {"ctrl_step": 3}, "good-params": {"ctrl_step": 2, "delay_step": 10}},
            },
            DelayStates(DelayStatesParams([2, 3, 4], ctrl_step=2, delay_step=10)),
        ),
        (
            {
                "name": "delay-states-all",
                "params": "good-params",
                "active_joints": [],
                "preview-ref": {"default": {"ctrl_step": 2}},
                "delay-states": {"default": {"ctrl_step": 3}, "good-params": {"ctrl_step": 2, "delay_step": 10}},
                "delay-states-all": {
                    "default": {"ctrl_step": 4},
                    "good-params": {"active_joints": [5], "ctrl_step": 5, "delay_step": 15},
                },
            },
            DelayStatesAll(DelayStatesAllParams([5], ctrl_step=5, delay_step=15)),
        ),
    ],
)
def test_load_data_adapter(config: dict[str, object], expected: DataAdapterBase) -> None:
    actual = load_data_adapter(config)
    assert type(actual) is type(expected)
    assert actual.params == expected.params


@pytest.mark.parametrize(
    "config",
    [
        {"name": "unknown_adapter", "active_joints": [0, 1]},
        {
            "name": "unknown_adapter",
            "params": "default",
            "active_joints": [1, 3],
            "preview-ref": {"default": {"ctrl_step": 3}},
        },
    ],
)
def test_load_data_adapter_error_unknown_name(config: dict[str, object]) -> None:
    msg = r"unknown data adapter name:"
    with pytest.raises(KeyError, match=msg):
        load_data_adapter(config)


@pytest.mark.parametrize(
    "config",
    [
        {
            "name": "delay-states",
            "params": "unknown-params",
            "active_joints": [2, 3, 4],
            "preview-ref": {"default": {"ctrl_step": 2}},
            "delay-states": {"default": {"ctrl_step": 3}, "good-params": {"ctrl_step": 2, "delay_step": 10}},
        },
    ],
)
def test_load_data_adapter_error_unknown_params_set(config: dict[str, object]) -> None:
    msg = r"unknown parameter set name:"
    with pytest.raises(KeyError, match=msg):
        load_data_adapter(config)


@pytest.mark.parametrize(
    "config",
    [
        {"active_joints": [0, 1]},
        {
            "params": "default",
            "active_joints": [1, 3],
            "preview-ref": {"default": {"ctrl_step": 3}},
        },
    ],
)
def test_load_data_adapter_error_no_name_key(config: dict[str, object]) -> None:
    msg = r"'name' is required to load data adapter"
    with pytest.raises(KeyError, match=msg):
        load_data_adapter(config)


@pytest.mark.parametrize(
    "config",
    [
        {"name": "delay-states", "active_joints": [0, 1, 2], "unknown_key": 0},
    ],
)
def test_load_data_adapter_error_unknown_key_in_params(config: dict[str, object]) -> None:
    msg = r"got an unexpected keyword argument"
    with pytest.raises(TypeError, match=msg):
        load_data_adapter(config)


SCALER_CONFIG_TEXT = """\
[model.scaler]
name = "std"
params = "default"

[model.scaler.std.default]
with_mean = true
with_std = true

[model.scaler.minmax.default]
feature_range = [0, 1]
clip = false

[model.scaler.maxabs.default]
[model.scaler.robust.default]
with_centering = true
with_scaling = true
quantile_range = [25.0, 75.0]
unit_variance = false
"""


@pytest.fixture
def scaler_config() -> dict:
    return tomllib.loads(SCALER_CONFIG_TEXT)


def test_load_scaler_typical_config(scaler_config: dict) -> None:
    config = scaler_config["model"]["scaler"]
    actual = load_scaler(config)
    expected = StandardScaler()
    assert actual is not None
    assert type(actual) is type(expected)
    assert actual.get_params() == expected.get_params()


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("minmax", MinMaxScaler()),
        ("std.default", StandardScaler()),
        ("robust.default", RobustScaler()),
        ("none", None),
    ],
)
def test_load_scaler_config_modified_by_user(
    scaler_config: dict,
    selector: str,
    expected: Scaler,
) -> None:
    config = scaler_config["model"]["scaler"]
    config = update_config_by_selector(config, selector)
    actual = load_scaler(config)
    if actual is not None:
        assert type(actual) is type(expected)
        assert actual.get_params() == expected.get_params()
    else:
        assert actual is None


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"name": "std"}, StandardScaler()),
        ({"name": "minmax", "feature_range": (0.1, 0.9)}, MinMaxScaler(feature_range=(0.1, 0.9))),
        ({"name": "maxabs"}, MaxAbsScaler()),
        ({"name": "robust", "with_centering": False}, RobustScaler(with_centering=False)),
        (
            {"name": "std", "params": "default", "std": {"default": {"with_std": False}}},
            StandardScaler(with_std=False),
        ),
        (
            {
                "name": "minmax",
                "params": "good-params",
                "std": {"default": {"with_std": False}},
                "minmax": {"default": {}, "good-params": {"feature_range": (0.2, 0.99), "clip": True}},
            },
            MinMaxScaler(feature_range=(0.2, 0.99), clip=True),
        ),
        (
            {
                "name": "robust",
                "params": "good-params",
                "std": {"default": {"with_std": False}},
                "minmax": {"default": {}, "good-params": {"feature_range": (0.2, 0.99), "clip": True}},
                "maxabs": {"default": {}},
                "robust": {
                    "default": {"with_centering": False},
                    "good-params": {"with_scaling": False, "quantile_range": (20.0, 80.0), "unit_variance": True},
                },
            },
            RobustScaler(with_scaling=False, quantile_range=(20.0, 80.0), unit_variance=True),
        ),
    ],
)
def test_load_scaler(config: dict[str, object], expected: Scaler) -> None:
    actual = load_scaler(config)
    assert actual is not None
    assert type(actual) is type(expected)
    assert actual.get_params() == expected.get_params()


@pytest.mark.parametrize(
    "config",
    [
        {"name": "none"},
        {"name": "None", "params": "default"},
        {
            "name": "NONE",
            "params": "good-params",
            "std": {"default": {"with_std": False}},
            "minmax": {"default": {}, "good-params": {"feature_range": (0.2, 0.99), "clip": True}},
            "maxabs": {"default": {}},
            "robust": {
                "default": {"with_centering": False},
                "good-params": {"with_scaling": False, "quantile_range": (20.0, 80.0), "unit_variance": True},
            },
        },
    ],
)
def test_load_scaler_none(config: dict[str, object]) -> None:
    actual = load_scaler(config)
    assert actual is None


@pytest.mark.parametrize(
    "config",
    [
        {"name": "unknown_scaler"},
        {"name": "unknown_scaler", "params": "default", "std": {"default": {"with_std": False}}},
    ],
)
def test_load_scaler_error_unknown_name(config: dict[str, object]) -> None:
    msg = r"unknown scaler name:"
    with pytest.raises(KeyError, match=msg):
        load_scaler(config)


@pytest.mark.parametrize(
    "config",
    [
        {
            "name": "minmax",
            "params": "unknown-params",
            "std": {"default": {"with_std": False}},
            "minmax": {"default": {}, "good-params": {"feature_range": (0.2, 0.99), "clip": True}},
        },
    ],
)
def test_load_scaler_error_unknown_params_set(config: dict[str, object]) -> None:
    msg = r"unknown parameter set name:"
    with pytest.raises(KeyError, match=msg):
        load_scaler(config)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"params": "default", "std": {"default": {}}},
    ],
)
def test_load_scaler_error_no_name_key(config: dict[str, object]) -> None:
    msg = r"'name' is required to load scaler"
    with pytest.raises(KeyError, match=msg):
        load_scaler(config)


@pytest.mark.parametrize(
    "config",
    [
        {"name": "std", "unknown_key": True},
    ],
)
def test_load_scaler_error_unknown_key_in_params(config: dict[str, object]) -> None:
    msg = r"got an unexpected keyword argument"
    with pytest.raises(TypeError, match=msg):
        load_scaler(config)


REGRESSOR_CONFIG_TEXT = """\
[model.regressor]
name = "mlp"
params = "default"

[model.regressor.linear.default]
fit_intercept = true
copy_X = true
n_jobs = "None"
positive = false

[model.regressor.ridge.default]
alpha = 1.0
fit_intercept = true
copy_X = true
max_iter = "None"
tol = 1e-4
solver = "auto"
positive = false
random_state = "None"

[model.regressor.mlp.default]
hidden_layer_sizes = [100]
activation = "relu"
solver = "adam"
alpha = 0.0001
batch_size = "auto"
learning_rate = "constant"
learning_rate_init = 0.001
power_t = 0.5
max_iter = 200
shuffle = true
random_state = "None"
tol = 1e-4
verbose = false
warm_start = false
momentum = 0.9
nesterovs_momentum = true
early_stopping = false
validation_fraction = 0.1
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8
n_iter_no_change = 10
"""


@pytest.fixture
def regressor_config() -> dict:
    return tomllib.loads(REGRESSOR_CONFIG_TEXT)


def test_load_regressor_typical_config(regressor_config: dict) -> None:
    config = regressor_config["model"]["regressor"]
    actual = load_regressor(config)
    expected = MLPRegressor()
    assert type(actual) is type(expected)
    assert actual.get_params() == expected.get_params()


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("linear", LinearRegression()),
        ("ridge.default", Ridge()),
        ("mlp.default", MLPRegressor()),
    ],
)
def test_load_regressor_config_modified_by_user(
    regressor_config: dict,
    selector: str,
    expected: Regressor,
) -> None:
    config = regressor_config["model"]["regressor"]
    config = update_config_by_selector(config, selector)
    actual = load_regressor(config)
    assert type(actual) is type(expected)
    assert actual.get_params() == expected.get_params()


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"name": "linear"}, LinearRegression()),
        (
            {"name": "ridge", "fit_intercept": False, "alpha": 0.9, "random_state": 42},
            Ridge(alpha=0.9, fit_intercept=False, random_state=42),
        ),
        (
            {"name": "mlp", "hidden_layer_sizes": [100, 100], "activation": "tanh", "batch_size": "auto"},
            MLPRegressor(hidden_layer_sizes=(100, 100), activation="tanh"),
        ),
        (
            {
                "name": "linear",
                "params": "good-params",
                "fit_intercept": False,
                "linear": {"default": {}, "good-params": {"fit_intercept": True, "n_jobs": -1}},
                "ridge": {"default": {"fit_intercept": False, "alpha": 0.9, "random_state": 42}},
            },
            LinearRegression(fit_intercept=True, n_jobs=-1),
        ),
        (
            {
                "name": "ridge",
                "params": "default",
                "fit_intercept": False,
                "linear": {"default": {}, "good-params": {"fit_intercept": True, "n_jobs": -1}},
                "ridge": {"default": {"fit_intercept": False, "max_iter": "None", "random_state": 42}},
            },
            Ridge(fit_intercept=False, max_iter=None, random_state=42),
        ),
        (
            {
                "name": "mlp",
                "params": "bad-params",
                "linear": {"default": {}},
                "ridge": {"default": {}},
                "mlp": {
                    "default": {},
                    "good-params": {
                        "hidden_layer_sizes": [200],
                        "activation": "logistic",
                        "max_iter": 200,
                        "random_state": 123,
                    },
                    "bad-params": {
                        "hidden_layer_sizes": [100, 100],
                        "activation": "relu",
                        "max_iter": 500,
                        "random_state": 123,
                    },
                },
            },
            MLPRegressor(hidden_layer_sizes=(100, 100), activation="relu", max_iter=500, random_state=123),
        ),
    ],
)
def test_load_regressor(config: dict[str, object], expected: Regressor) -> None:
    actual = load_regressor(config)
    assert actual is not None
    assert type(actual) is type(expected)
    assert actual.get_params() == expected.get_params()


@pytest.mark.parametrize(
    "config",
    [
        {"name": "unknown_regressor"},
        {"name": "unknown_regressor", "params": "default", "linear": {"default": {"fit_intercept": False}}},
    ],
)
def test_load_regressor_error_unknown_name(config: dict[str, object]) -> None:
    msg = r"unknown regressor name:"
    with pytest.raises(KeyError, match=msg):
        load_regressor(config)


@pytest.mark.parametrize(
    "config",
    [
        {
            "name": "ridge",
            "params": "unknown-params",
            "linear": {"default": {"fit_intercept": False}},
            "ridge": {"default": {}, "good-params": {"fit_intercept": False, "alpha": 0.1, "random_state": 42}},
        },
    ],
)
def test_load_regressor_error_unknown_params_set(config: dict[str, object]) -> None:
    msg = r"unknown parameter set name:"
    with pytest.raises(KeyError, match=msg):
        load_regressor(config)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"params": "default", "linear": {"default": {}}},
    ],
)
def test_load_regressor_error_no_name_key(config: dict[str, object]) -> None:
    msg = r"'name' is required to load regressor"
    with pytest.raises(KeyError, match=msg):
        load_regressor(config)


@pytest.mark.parametrize(
    "config",
    [
        {"name": "linear", "unknown_key": True},
    ],
)
def test_load_regressor_error_unknown_key_in_params(config: dict[str, object]) -> None:
    msg = r"got an unexpected keyword argument"
    with pytest.raises(TypeError, match=msg):
        load_regressor(config)


def main() -> None:
    import sys

    start_event_logging(sys.argv, logging_level="INFO")
    msg = f"""\
    Provide a number to generate expected data.

    MENU:
      1) Expected data for testing SimpleDataAdapter class
      2) Expected data for testing JointDataAdapter class

    EXAMPLE:
      $ python {' '.join(sys.argv)} 1
"""
    if len(sys.argv) > 1:
        match sys.argv[1]:
            case "1":
                generate_expected_data_for_simple_data_adapter(show_plot=True)
            case "2":
                generate_expected_data_for_joint_data_adapter(show_plot=True)
            case _:
                raise RuntimeError(msg)
    else:
        raise RuntimeError(msg)


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "Ctrl adam arg cb csv ctrl dq init iter maxabs minmax mlp nesterovs noqa params pb pred qdes quantile quntile regressor relu scaler sgd sklearn tanh tol" # noqa: E501
# End:
