from abc import ABC, abstractmethod
from typing import Self

import xarray as xr
import numpy as np

from data_downloading.datasets import MortalityDataset


class Model(ABC):
    def __init__(
            self,
            lee_miller_fix: bool = False,
            seed: int | np.random.Generator | None = None
        ) -> None:
        self.lee_miller_fix = lee_miller_fix
        self.seed = seed

    @abstractmethod
    def fit(self, mortality_data: MortalityDataset, value_column: str) -> Self:
        """Fit the model on the mortality data with a specified value column,
        using an individually set up method.

        Parameters
        ----------
        mortality_data
            Instance of the MortalityDataset, with loaded data depending on
            the model architecture, similar to 'x' in sklearn.
        value_column
            The chosen value column used for fitting the model,
            similar to 'y' in sklearn.
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
            Data forecast (either a matrix or a tensor,
            depending on the number of simulations chosen).
        """
        pass

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

    def _validate_dataset(
            self, 
            mortality_data: MortalityDataset, 
            required_grids: list[str]
        ) -> None:
        """Check if the specified datasets are present in the MortalityDataset
        instance (models themselves dictate what they want to check). 
        
        If there is more than one required grid to be checked, this method also
        validates that every grid contains the exact same timespan.

        Parameters
        ----------
        mortality_data
            Instance of the MortalityDataset.
        required_grids
            The model specified grids this method has to check.
        """
        for grid in required_grids:
            selected_grid = getattr(mortality_data, grid, None)
            if (
                selected_grid is None or 
                selected_grid.data is None or 
                selected_grid.data.empty
            ):
                raise ValueError(
                    f"The necessary grid '{grid}' for this model is " \
                    f"not available in the mortality data."
                )

        if len(required_grids) > 1:
            reference_year_interval = getattr(
                mortality_data, 
                required_grids[0]
            ).year_interval

            for grid in required_grids:
                if reference_year_interval != getattr(mortality_data, grid).year_interval:
                    raise ValueError(
                        f"Year interval mismatch between grid " \
                        f"'{required_grids[0]}' and grid '{grid}'"
                    )
        
