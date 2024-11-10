from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyplotutil.datautil import Data
from sklearn import datasets

from affetto_nn_ctrl.model_utility import DataAdapter, DataAdapterParams, Reference


@dataclass
class SimpleDataAdapterParams(DataAdapterParams):
    feature_index: list[int]


class SimpleDataAdapter:
    def __init__(self, params: SimpleDataAdapterParams) -> None:
        self.params = params

    def to_feature(self, **dataset: np.ndarray) -> np.ndarray:
        dataset[""]

    def to_target(self, **dataset: np.ndarray) -> np.ndarray:
        pass

    def make_features(self, dataset: Data) -> np.ndarray:
        keys = dataset.keys()
        for values in dataset:
            x = self.to_feature(**dict(zip(keys, values, strict=True)))

    def make_target(self, dataset: Data) -> np.ndarray:
        keys = dataset.keys()
        for values in dataset:
            x = self.to_feature(**dict(zip(keys, values, strict=True)))

    def make_model_input(self, **states: np.ndarray | Reference) -> np.ndarray:
        pass

    def make_ctrl_input(self, y: np.ndarray, **base_input: np.ndarray) -> tuple[np.ndarray, ...]:
        pass


def make_dataset(n_samples: int, n_features: int, n_targets: int) -> Data:
    X, y = datasets.make_regression(  # type: ignore[] # noqa: N806
        n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_targets=n_targets,
        bias=100,
        noise=4.0,
    )
    data = np.hstack((X, y))
    columns = [f"f{x}" for x in range(n_features)] + [f"t{x}" for x in range(n_targets)]
    return Data(pd.DataFrame(dict(zip(columns, data.T, strict=True))))


# Local Variables:
# jinx-local-words: "noqa"
# End:
