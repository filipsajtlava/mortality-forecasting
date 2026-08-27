from abc import ABC, abstractmethod

import numpy as np
import xarray as xr


class Forecaster(ABC):
    def __init__(self, seed: int | np.random.Generator | None = None) -> None:
        self.seed = seed

    def _normalize_seed(self) -> np.random.Generator:
        """Normalizes the entered seed into a single np.random.Generator instance

        Returns
        -------
        np.random.Generator
            An active NumPy random number generator instance.
        """
        if isinstance(self.seed, np.random.Generator):
            return self.seed
        return np.random.default_rng(self.seed)

    @abstractmethod
    def fit(self, parameter_dataset: xr.Dataset) -> None:
        pass

    @abstractmethod
    def forecast_parameters(self, steps: int, simulations: int) -> xr.Dataset:
        pass