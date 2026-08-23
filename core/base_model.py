from abc import ABC, abstractmethod
from typing import Self

import xarray as xr

from data_downloading.datasets import MortalityDataset
from core.base_forecaster import Forecaster, DualForecaster
from core.commons import ForecastContainer, ParameterContainer
from plotting.model_plot import ModelPlotter


class Model(ABC):
    # TODO: since the model plotters depend on parameters_, that should be explicitely
    # implemented as an abstract method - property, so that it enforces the idea that
    # every model has to have it, making the plotter unbreakable
    # TODO: if other models wont accept the lee_miller fix (only the lc will), its good to remove
    # it from here and just put it individually into the submodel init, calling super().__init__(seed)
    def __init__(
            self,
            lee_miller_fix: bool = False
        ) -> None:
        self.lee_miller_fix = lee_miller_fix

    @property
    def plot(self) -> ModelPlotter:
        return ModelPlotter(self)

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
    def _predict_mortalities(
            self, 
            forecasted_values: ParameterContainer
        ) -> xr.DataArray:
        pass

    def predict_in_sample(self) -> xr.DataArray:
        """Predicts the mortalities from the fitted parameters."""
        self._check_if_fitted()
        return self._predict_mortalities(self.parameters_)

    def forecast(
            self, 
            forecaster: Forecaster | DualForecaster,
            steps: int, 
            simulations: int = 1
        ) -> ForecastContainer:

        param_container = self._forecast_parameters(
            forecaster, 
            steps, 
            simulations
        )
        predicted_mortalities = self._predict_mortalities(param_container)
        return ForecastContainer(predicted_mortalities, param_container)

    def _forecast_parameters(
            self, 
            forecaster: Forecaster | DualForecaster,
            steps: int, 
            simulations: int = 1
        ) -> ParameterContainer:
        self._check_if_fitted()

        # TODO: I genuinely dislike how this is done, other way of approaching
        # this would be by calling a different function, something like
        # fit_forecast, that does both at once, and has different definitions
        # for DualForecaster and Forecaster, so they handle it internally, out
        # of the model
        if isinstance(forecaster, Forecaster):
            forecaster.fit(self.parameters_.period)
            period_ds = forecaster.forecast_parameters(steps, simulations)
            return ParameterContainer(
                static=self.parameters_.static,
                period=period_ds
            )
        elif isinstance(forecaster, DualForecaster):
            forecaster.fit(self.parameters_)
            period_ds, cohort_ds = forecaster.forecast_parameters(steps, simulations)
            return ParameterContainer(
                static=self.parameters_.static,
                period=period_ds,
                cohort=cohort_ds
            )
        raise ValueError("Please enter a valid forecaster instance.")

    # TODO: parameters_ arent enforced everywhere else, so its kind-of weird
    # to be expecting every model to automatically have them (IT SHOULD BE ENFORCED)
    # TODO: The @attribute approach is bad, doesnt really enforce it, as I always
    # have to add the basically empty attribute method parameters_, but after
    # that I still have to define the parameters_ individually in the fit
    # TODO: Thats also one of the problems, some centralisation of all the models
    # and what their estimated parameters are would be nice, like dictionaries
    # of static, period and cohort, along with their names.
    def _check_if_fitted(self) -> None:
        parameters = getattr(self, "parameters_", None)
        if parameters is None:
            raise ValueError("You need to fit the model first.")

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
            # TODO: This is checking for way too much, the grids themselves should
            # never allow for adding of empty datasets, for example
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
                        f"'{required_grids[0]}' and grid '{grid}'."
                    )