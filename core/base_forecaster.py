from abc import ABC, abstractmethod

import numpy as np
import xarray as xr

from core.commons import ParameterContainer


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


class DualForecaster:
    def __init__(self, period: Forecaster, cohort: Forecaster) -> None:
        self.period = period
        self.cohort = cohort

    def fit(self, parameter_container: ParameterContainer) -> None:
        if parameter_container.cohort is None:
            raise ValueError(
                "This model is incompatible with a dual forecaster, " \
                "as it does not contain a cohort component."
            )

        self.period.fit(parameter_container.period)
        self.cohort.fit(parameter_container.cohort)

    def forecast_parameters(
            self, 
            steps: int, 
            simulations: int
        ) -> tuple[xr.Dataset, xr.Dataset]:
        period_forecasted = self.period.forecast_parameters(steps, simulations)
        cohort_forecasted = self.cohort.forecast_parameters(steps, simulations)
        return period_forecasted, cohort_forecasted
    