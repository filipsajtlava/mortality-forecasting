from abc import ABC, abstractmethod
from typing import Self, ClassVar, Literal

import xarray as xr
import numpy as np

from mortality_forecasting import config
from mortality_forecasting.core._commons import ForecastContainer, ParameterContainer
from mortality_forecasting.core._base_forecaster import Forecaster
from mortality_forecasting.forecasting._dual_forecaster import DualForecaster
from mortality_forecasting.forecasting._bootstrapper import Bootstrapper
from mortality_forecasting.models.single._parameters import ModelParams
from mortality_forecasting.models._likelihood_families import Poisson, NegativeBinomial
from mortality_forecasting.plotting._model_plot import ModelPlotter


class Model(ABC):
    _PARAMS: ClassVar[ModelParams]
    FAMILY_MAP = {
        "poisson": Poisson(),
        "negative_binomial": NegativeBinomial() 
    }

    def __init__(
            self,
            method: Literal["poisson", "negative_binomial"] | str,
            num_initialization: Literal["naive", "SVD"] | ParameterContainer = "SVD",
            lee_miller_fix: bool = False,
            ftol: float = 1e-7,
            verbose: bool = False
        ) -> None:
        self.method = method
        self.num_initialization = num_initialization
        self.lee_miller_fix = lee_miller_fix
        self.ftol = ftol
        self.verbose = verbose

        self.family = self.FAMILY_MAP.get(method)

    @property
    def plot(self) -> ModelPlotter:
        return ModelPlotter(self)

    def _check_if_fitted(self) -> None:
        parameters = getattr(self, "parameters_", None)
        if parameters is None:
            raise ValueError("You need to fit the model first.")

    @abstractmethod
    def _fit(self):
        pass

    @abstractmethod
    def _compute_deaths(self, **kwargs) -> xr.DataArray:
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

    def fit(self, mortality_data: xr.Dataset) -> Self:
        self.mortality_data = self._validate_dataset(mortality_data)
        self._bound_dims = {
            dim: self.mortality_data.coords[dim].values
            for dim in self.mortality_data.dims
            if dim not in (config.AGE_DIM, config.YEAR_DIM)
        }
        self.D = self.mortality_data.D
        self.E = self.mortality_data.E
        self.M = self.D / self.E
        self._year_coords = self.D.coords[config.YEAR_DIM]
        self._age_coords = self.D.coords[config.AGE_DIM]

        estimated_params = self._fit()
        self.parameters_ = self._PARAMS._build_container(estimated_params)
        if not (estimated_params.get("lambda_dispersion", xr.DataArray([None])) == None).any():
            self.parameters_.static["lambda_dispersion"] = estimated_params["lambda_dispersion"]
        return self

    def _initialize_parameters(self) -> dict[str, xr.DataArray | None]:
        initialized_params = {}
        lambda_init = None

        if self.num_initialization == "SVD":
            from mortality_forecasting.models.single._lee_carter import LeeCarterModel  
            lc_model = LeeCarterModel(method=self.num_initialization).fit(self.mortality_data)
            for param_spec in self._PARAMS:
                if lc_model.parameters_.get(param_spec.name, None) is not None:
                    initialized_params[param_spec.name] = lc_model.parameters_[param_spec.name]
                
        elif isinstance(self.num_initialization, ParameterContainer):
            for param_spec in self._PARAMS:
                initialized_params[param_spec.name] = self.num_initialization.get(
                    param_spec.name, None
                )
            lambda_init = self.num_initialization.get("lambda_dispersion", None)

        for param_spec in self._PARAMS:
            if param_spec.name not in initialized_params.keys():
                dims = [param_spec.dimension] + list(self._bound_dims.keys())
                initialized_params[param_spec.name] = xr.DataArray(
                    param_spec.naive_value, 
                    dims=dims,
                    coords={
                        **self._bound_dims,
                        param_spec.dimension: (
                            np.unique(self._year_coords - self._age_coords)
                            if param_spec.dimension == config.COHORT_DIM 
                            else self.D.coords[param_spec.dimension].values
                        )
                    }
                )
        
        initialized_params["lambda_dispersion"] = self.family.dispersion_initialization(
            bound_dims=self._bound_dims,
            init=lambda_init
        )
        return initialized_params

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
            attrs={"alpha": alpha, "bound_dims": self._bound_dims}
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