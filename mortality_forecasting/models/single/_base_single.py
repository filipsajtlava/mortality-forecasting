from abc import ABC, abstractmethod
from typing import Self, ClassVar

import xarray as xr

from mortality_forecasting.core._base_model import Model
from mortality_forecasting import config
from mortality_forecasting.core._commons import ForecastContainer, ParameterContainer
from mortality_forecasting.core._base_forecaster import Forecaster
from mortality_forecasting.forecasting._dual_forecaster import DualForecaster
from mortality_forecasting.forecasting._bootstrapper import Bootstrapper
from mortality_forecasting.models.single._parameters import ModelParams


class SinglePopulationModel(Model, ABC):
    _MODEL_PARAMS: ClassVar[ModelParams]

    @abstractmethod
    def _fit(self):
        pass

    def fit(self, mortality_data: xr.Dataset) -> Self:
        self.mortality_data = self._validate_dataset(mortality_data)
        self._bound_dims = {
            dim: self.mortality_data.coords[dim].values
            for dim in self.mortality_data.dims
            if dim not in (config.AGE_DIM, config.YEAR_DIM)
        }
        self._bound_dims
        self.D = self.mortality_data.D
        self.E = self.mortality_data.E
        self.M = self.D / self.E

        self._fit()
        return self

    def forecast(
            self, 
            forecaster: Forecaster | DualForecaster,
            steps: int,
            alpha: float = 0.05,
            bootstrapper: Bootstrapper | None = None
        ) -> ForecastContainer:
        self._check_if_fitted()

        if isinstance(bootstrapper, Bootstrapper):
            return bootstrapper.run(
                model=self, 
                forecaster=forecaster, 
                steps=steps,
                alpha=alpha
            )

        if isinstance(forecaster, Forecaster):
            forecaster.fit(self.parameters_.period)
            period_ds = forecaster.forecast_parameters(steps, alpha)
            parameters = ParameterContainer(
                static=self.parameters_.static,
                period=period_ds
            )
        elif isinstance(forecaster, DualForecaster):
            forecaster.fit(self.parameters_, alpha)
            period_ds, cohort_ds = forecaster.forecast_parameters(steps, alpha)
            parameters = ParameterContainer(
                static=self.parameters_.static,
                period=period_ds,
                cohort=cohort_ds
            )
        else:
            raise ValueError("Please enter a valid forecaster instance.")

        predicted_mortalities = self._predict_mortalities(parameters)
        return ForecastContainer(
            mortality_rates_=predicted_mortalities,
            parameters_=parameters, 
            attrs={"alpha": alpha}
        )

    def _validate_dataset(
            self,
            mortality_data: xr.Dataset
        ) -> xr.Dataset:        
        reference_grid = None
        for grid in config.REQUIRED_DATA_FOR_MODELS:
            selected_grid = getattr(mortality_data, grid, None)
            if selected_grid is None:
                raise ValueError(
                    f"Dataset does not contain the grid '{grid}'."
                )

            # TODO: this year-interval mismatch checker could be moved to commons,
            # and maybe used in the manual loader, to look if the years are the same
            if reference_grid is None:
                reference_grid = selected_grid
            self._year_interval_mismatch_check(reference_grid, selected_grid)
        return mortality_data

    def _year_interval_mismatch_check(
            self,
            reference_da: xr.DataArray,
            selected_da: xr.DataArray
        ) -> None:
            reference_grid_year_interval = (
                reference_da.coords[config.YEAR_DIM].min(),
                reference_da.coords[config.YEAR_DIM].max()
            )
            selected_grid_year_interval = (
                selected_da.coords[config.YEAR_DIM].min(),
                selected_da.coords[config.YEAR_DIM].max()
            )
            if reference_grid_year_interval != selected_grid_year_interval:
                raise ValueError(
                    f"Year interval mismatch between individual subgrids"
                )

    @property
    @abstractmethod
    def deviance_residuals(self) -> xr.DataArray:
        pass

    @property
    @abstractmethod
    def pearson_residuals(self) -> xr.DataArray:
        pass