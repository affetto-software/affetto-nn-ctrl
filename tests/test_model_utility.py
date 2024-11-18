from __future__ import annotations

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
from affetto_nn_ctrl.event_logging import start_event_logging
from affetto_nn_ctrl.model_utility import DataAdapterParams, Reference, Regressor

try:
    from . import TESTS_DATA_DIR_PATH, assert_file_contents
except ImportError:
    import sys

    sys.path.append(str(ROOT_DIR_PATH))
    from tests import TESTS_DATA_DIR_PATH, assert_file_contents  # type: ignore[reportMissingImports]

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class SimpleDataAdapterParams(DataAdapterParams):
    feature_index: list[int]
    target_index: list[int]


class SimpleDataAdapter:
    def __init__(self, params: SimpleDataAdapterParams) -> None:
        self.params = params

    def make_feature(self, dataset: Data) -> np.ndarray:
        feature_keys = [f"f{x}" for x in self.params.feature_index]
        feature_data = dataset.dataframe[feature_keys]
        return feature_data.to_numpy()

    def make_target(self, dataset: Data) -> np.ndarray:
        target_keys = [f"t{x}" for x in self.params.target_index]
        target_data = dataset.dataframe[target_keys]
        return target_data.to_numpy()

    def make_model_input(self, t: float, **states: np.ndarray | Reference) -> np.ndarray:
        inputs: list[float] = []
        for i in self.params.feature_index:
            key = f"f{i}"
            values = states[key]
            if isinstance(values, np.ndarray):
                inputs.extend(values)
            elif isinstance(values, Callable):  # type: ignore[arg-type]
                # https://github.com/python/mypy/issues/3060
                inputs.extend(values(t))
            else:
                msg = f"unsupported type: states[{key}] = {values} ({type(values)})"
                raise TypeError(msg)
        return np.asarray([inputs], dtype=float)

    def make_ctrl_input(self, y: np.ndarray, **base_input: np.ndarray) -> tuple[np.ndarray, ...]:
        _ = base_input
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = np.ravel(y)
        if not isinstance(y, Iterable):
            return (y,)
        return tuple(y)

    def reset(self) -> None:
        pass


def make_dataset(n_samples: int, n_features: int, n_targets: int) -> tuple[Data, Data]:
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


def train_model(train_dataset: Data, adapter: SimpleDataAdapter, model: Regressor) -> Regressor:
    x_train = adapter.make_feature(train_dataset)
    y_train = adapter.make_target(train_dataset)
    if len(adapter.params.target_index) == 1:
        y_train = np.ravel(y_train)
    return model.fit(x_train, y_train)


def predict(test_dataset: Data, adapter: SimpleDataAdapter, model: Regressor) -> np.ndarray:
    prediction: list[tuple[np.ndarray, ...]] = []
    for x_input in test_dataset:
        kw = dict(zip(test_dataset.columns, map(np.atleast_1d, x_input), strict=True))
        t = 0
        x = adapter.make_model_input(t, **kw)
        y = model.predict(x)
        c = adapter.make_ctrl_input(np.asarray(y))
        prediction.append(c)
    return np.asarray(prediction)


def make_prediction_data_path(base_directory: Path, model_name: str, **model_params: str | float) -> Path:
    filename = model_name
    if len(model_params):
        joined = "_".join("-".join(map(str, x)) for x in model_params.items())
        filename += f"_{joined}"
    filename += ".csv"
    return base_directory / filename


def generate_prediction_data(
    output: Path,
    train_dataset: Data,
    test_dataset: Data,
    adapter: SimpleDataAdapter,
    model: Regressor,
) -> np.ndarray:
    model = train_model(train_dataset, adapter, model)
    prediction = predict(test_dataset, adapter, model)
    np.savetxt(output, prediction)
    return prediction


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
def test_regression_models(make_work_directory: Path, model: Regressor, kw: dict[str, str | float], name: str) -> None:
    train_dataset, test_dataset = make_dataset(n_samples=20, n_features=1, n_targets=1)
    adapter = SimpleDataAdapter(SimpleDataAdapterParams(feature_index=[0], target_index=[0]))
    output = make_prediction_data_path(make_work_directory, name, **kw)
    generate_prediction_data(output, train_dataset, test_dataset, adapter, model)
    assert_file_contents(TESTS_DATA_DIR_PATH / output.name, output)


def check_expected_data(
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


def generate_expected_data(*, show_plot: bool = True) -> None:
    train_dataset, test_dataset = make_dataset(n_samples=20, n_features=1, n_targets=1)
    adapter = SimpleDataAdapter(SimpleDataAdapterParams(feature_index=[0], target_index=[0]))

    models: list[tuple[Regressor, dict[str, str | float], str]] = [
        (LinearRegression(), {}, "linear"),
        (Ridge(alpha=0.1), {"alpha": 0.1}, "ridge"),
        (Ridge(alpha=1.0), {"alpha": 1.0}, "ridge"),
        (MLPRegressor(random_state=42, max_iter=200), {"max_iter": 200}, "mlp"),
        (MLPRegressor(random_state=42, max_iter=500), {"max_iter": 500}, "mlp"),
        (MLPRegressor(random_state=42, max_iter=800), {"max_iter": 800}, "mlp"),
    ]
    for model, kw, name in models:
        output = make_prediction_data_path(TESTS_DATA_DIR_PATH, name, **kw)
        prediction = generate_prediction_data(output, train_dataset, test_dataset, adapter, model)
        if show_plot:
            check_expected_data(train_dataset, test_dataset, adapter, prediction, output.stem)


def main() -> None:
    import sys

    start_event_logging(sys.argv, logging_level="INFO")
    generate_expected_data(show_plot=True)


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "arg csv iter mlp noqa params"
# End:
