from abc import ABC, abstractmethod
from typing import Literal

import xarray as xr

from mortality_forecasting.forecasting._dual_forecaster import DualForecaster
from ._base_forecaster import Forecaster
from ._commons import ForecastContainer, ParameterContainer
from mortality_forecasting.plotting._model_plot import ModelPlotter
from mortality_forecasting.models._likelihood_families import Poisson, NegativeBinomial


class Model(ABC):
    FAMILY_MAP = {
        "poisson": Poisson(),
        "negative_binomial": NegativeBinomial() 
    }

    # TODO: if other models wont accept the lee_miller fix (only the lc will), its good to remove
    # it from here and just put it individually into the submodel init, calling super().__init__(seed)
    def __init__(
            self,
            method: Literal["poisson", "negative_binomial"] | str,
            lee_miller_fix: bool
        ) -> None:
        self.method = method
        self.lee_miller_fix = lee_miller_fix
        self.family = self.FAMILY_MAP.get(method)

    # TODO: Some centralisation of all the models
    # and what their estimated parameters are would be nice, like dictionaries
    # of static, period and cohort, along with their names.
    def _check_if_fitted(self) -> None:
        parameters = getattr(self, "parameters_", None)
        if parameters is None:
            raise ValueError("You need to fit the model first.")

    @property
    def plot(self) -> ModelPlotter:
        return ModelPlotter(self)

    @property
    @abstractmethod
    def parameters_(self) -> ParameterContainer:
        pass

    @abstractmethod
    def _predict_mortalities(
            self, 
            forecasted_values: ParameterContainer
        ) -> xr.DataArray:
        pass

    @abstractmethod
    def bootstrap(self):
        pass

    def predict_in_sample(self) -> xr.DataArray:
        """Predicts the mortalities from the fitted parameters."""
        self._check_if_fitted()
        return self._predict_mortalities(self.parameters_)

    def forecast(
            self, 
            forecaster: Forecaster | DualForecaster,
            steps: int
        ) -> ForecastContainer:
        self._check_if_fitted()

        # TODO: I genuinely dislike how this is done, other way of approaching
        # this would be by calling a different function, something like
        # fit_forecast, that does both at once, and has different definitions
        # for DualForecaster and Forecaster, so they handle it internally, out
        # of the model
        # TODO: just polymorph this I cba, redo forecasters internally and have
        # this like a super simple interface
        if isinstance(forecaster, Forecaster):
            forecaster.fit(self.parameters_.period)
            period_ds = forecaster.forecast_parameters(steps)
            parameters = ParameterContainer(
                static=self.parameters_.static,
                period=period_ds
            )
        elif isinstance(forecaster, DualForecaster):
            forecaster.fit(self.parameters_)
            period_ds, cohort_ds = forecaster.forecast_parameters(steps)
            return ParameterContainer(
                static=self.parameters_.static,
                period=period_ds,
                cohort=cohort_ds
            )
        else:
            raise ValueError("Please enter a valid forecaster instance.")

        predicted_mortalities = self._predict_mortalities(parameters)

        return ForecastContainer(predicted_mortalities, parameters)