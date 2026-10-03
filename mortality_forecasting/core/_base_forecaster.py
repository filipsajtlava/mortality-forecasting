from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import xarray as xr


class Forecaster(ABC):
    def __init__(
            self, 
            seed: int | np.random.Generator | None = None,
            simulations: int | None = None,
            return_simulations: bool = False,
            point_estimate: Literal["mean", "median"] = "median"
        ) -> None:
        self.seed = self._normalize_seed(seed)
        self.simulations = simulations
        self.return_simulations = return_simulations
        self.point_estimate =point_estimate

    def _normalize_seed(self, seed: int | np.random.Generator | None) -> np.random.Generator:
        """Normalizes the entered seed into a single np.random.Generator instance

        Returns
        -------
        np.random.Generator
            An active NumPy random number generator instance.
        """
        if isinstance(seed, np.random.Generator):
            return seed
        return np.random.default_rng(seed)

    @abstractmethod
    def fit(self, parameter_dataset: xr.Dataset) -> None:
        pass

    @abstractmethod
    def forecast_parameters(self, steps: int, alpha: float) -> xr.Dataset:
        pass