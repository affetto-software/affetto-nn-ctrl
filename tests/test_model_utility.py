from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyplotutil.datautil import Data
from sklearn import datasets
from sklearn.neural_network import MLPRegressor

from affetto_nn_ctrl.model_utility import DataAdapter, DataAdapterParams, Reference


@dataclass
class SimpleDataAdapterParams(DataAdapterParams):
    feature_index: list[int]
    target_index: list[int]


class SimpleDataAdapter:
    def __init__(self, params: SimpleDataAdapterParams) -> None:
        self.params = params

    def make_features(self, dataset: Data) -> np.ndarray:
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
        return tuple(y)


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


# Local Variables:
# jinx-local-words: "noqa"
# End:
