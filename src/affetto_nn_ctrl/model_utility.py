from __future__ import annotations

from typing import Callable

import numpy as np
from affctrllib import Logger


class DataAdapterParams:
    pass


class DataAdapter:
    def __init__(self, params: DataAdapterParams) -> None:
        pass

    def map_to_train_input(self, logger: Logger) -> np.ndarray:
        x: np.ndarray = np.array(())
        return x

    def map_to_train_output(self, logger: Logger) -> np.ndarray:
        y: np.ndarray = np.array(())
        return y

    def map_to_predict_input(
        self,
        logger: Logger,
        qdes_f: Callable[[float], np.ndarray],
        deqdes_f: Callable[[float], np.ndarray],
    ) -> np.ndarray:
        x: np.ndarray = np.array(())
        return x

    def map_predicted_output(self, y: np.ndarray) -> np.ndarray:
        c: np.ndarray = np.array(())
        return c
