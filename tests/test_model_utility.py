from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from pyplotutil.datautil import Data
from sklearn import datasets
from sklearn.neural_network import MLPRegressor

from affetto_nn_ctrl.event_logging import start_event_logging
from affetto_nn_ctrl.model_utility import DataAdapter, DataAdapterParams, Reference


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
            values = states[f"f{i}"]
            if isinstance(values, np.ndarray):
                inputs.extend(values)
            else:
                inputs.extend(values(t))
        return np.asarray(inputs, dtype=float)

    def make_ctrl_input(self, y: np.ndarray, **base_input: np.ndarray) -> tuple[np.ndarray, ...]:
        _ = base_input
        if not isinstance(y, Iterable):
            return (y,)
        return tuple(y)

    def reset(self) -> None:
        pass


def make_dataset(n_samples: int, n_features: int, n_targets: int) -> Data:
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
    return Data(pd.DataFrame(dict(zip(columns, data.T, strict=True))))


def fit_mlp_with_simple_data_adapter(train_dataset: Data, test_dataset: Data, params: SimpleDataAdapterParams) -> None:
    adapter = SimpleDataAdapter(params)
    x_train = adapter.make_feature(train_dataset)
    y_train = adapter.make_target(train_dataset)
    if len(params.target_index) == 1:
        y_train = np.ravel(y_train)
    regr = MLPRegressor(random_state=42, max_iter=500).fit(x_train, y_train)
    prediction: list[tuple[np.ndarray, ...]] = []
    for x_input in test_dataset:
        kw = dict(zip(test_dataset.columns, x_input, strict=True))
        t = 0
        x = adapter.make_model_input(t, **kw)
        y = np.asarray(regr.predict(x))
        c = adapter.make_ctrl_input(y)
        prediction.append(c)


def main() -> None:
    import sys

    start_event_logging(sys.argv, logging_level="INFO")

    dataset = make_dataset(n_samples=20, n_features=6, n_targets=4)
    params = SimpleDataAdapterParams(feature_index=[0, 2, 4], target_index=[1, 3])
    train_dataset, test_dataset = dataset.split_by_row(int(0.75 * 20))
    fit_mlp_with_simple_data_adapter(train_dataset, test_dataset, params)


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "noqa"
# End:
