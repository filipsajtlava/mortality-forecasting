import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod
from core.data_structures import MortalityData

class Model(ABC):
    def __init__(self, mortality_dataclass: MortalityData, value_column: str) -> None:
        self.mortality_dataclass = mortality_dataclass
        self.value_column = value_column
        self.wide_matrix = self.mortality_dataclass.get_pivoted_data(self.value_column)

    @abstractmethod
    def fit(self):
        """Fit the model using the specified method.
        """
        pass

    @abstractmethod
    def predict(self, steps: int, simulations: int = 1) -> xr.DataArray:
        """Forecast the future values from the fitted model.

        Parameters
        ----------
        steps
            Amount of years forecast into the future.
        simulations, optional
            Number of simulations of the stochastic forecasts, by default 1.

        Returns
        -------
            Data forecast (either a matrix or a tensor depending on the number of simulations chosen).
        """
        pass