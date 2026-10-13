from statistics import NormalDist
from typing import Literal

import numpy as np
import xarray as xr

from mortality_forecasting.core._base_forecaster import Forecaster
from mortality_forecasting import config
from mortality_forecasting.core._commons import bounds_from_simulations


class RandomWalkWithDrift(Forecaster):
    def __init__(
            self, 
            seed: int | np.random.Generator | None = None,
            simulations: int | None = None,
            return_simulations: bool = False,
            point_estimate: Literal["mean", "median"] = "median",
            overlap: bool = False
        ) -> None:
        super().__init__(
            seed=seed,
            simulations=simulations,
            return_simulations=return_simulations,
            point_estimate=point_estimate,
            overlap=overlap
        )

    # TODO: keeping the outputs in dictionaries for now, more complex forecasters
    # may require special output containers, just like ParameterContainer, dont forget
    # to change this approach as well after that!
    def fit(self, parameter_dataset: xr.Dataset) -> None:
        self.parameter_dataset = parameter_dataset
        self.parameters_for_forecasting_ = {}

        for parameter_name, parameter_da in self.parameter_dataset.items():
            param_main_dim = parameter_da.dims[0]
            drift_ = parameter_da[-1] - parameter_da[0] / (len(parameter_da) - 1)
            std_of_errors_ = ((
                parameter_da - drift_
            ).diff(dim=param_main_dim)).std(dim=param_main_dim, ddof=1)   
            self.parameters_for_forecasting_[parameter_name] = {
                "drift_": drift_,
                "std_of_errors_": std_of_errors_
            }

    def forecast_parameters(self, steps: int, alpha: float) -> xr.Dataset:
        self.alpha = alpha
        forecasted_parameters = xr.Dataset()
        last_year = int(self.parameter_dataset.coords[config.YEAR_DIM].max())
        pred_years = np.arange(0, steps + 1) + last_year

        for parameter_name, estimates in self.parameters_for_forecasting_.items():
            if self.simulations is not None:
                forecasts_da = self._forecast_parameter_stochastic(
                    parameter_name,
                    estimates["drift_"],
                    estimates["std_of_errors_"],
                    pred_years,
                )
            else:
                forecasts_da = self._forecast_parameter_analytical(
                    parameter_name,
                    estimates["drift_"],
                    estimates["std_of_errors_"],
                    pred_years
                )
            forecasted_parameters[parameter_name] = forecasts_da
        return forecasted_parameters

    def _forecast_parameter_stochastic(
            self, 
            parameter_name: str,
            estimated_drift: xr.DataArray,
            estimated_std_of_errors: xr.DataArray,
            pred_years: np.ndarray,
        ) -> xr.DataArray:
        raw_noise = xr.DataArray(
            self.seed.normal(
                size=(len(pred_years), self.simulations) + estimated_drift.shape
            ),
            dims=(config.YEAR_DIM, config.SIMULATION_DIM, *estimated_drift.dims),
            coords={
                config.YEAR_DIM: pred_years,
                config.SIMULATION_DIM: np.arange(1, self.simulations + 1),
                **estimated_drift.coords
            }
        )
        innovations = (
            estimated_drift + raw_noise * estimated_std_of_errors
        )

        last_value = (
            self.parameter_dataset[parameter_name]
            .isel({config.YEAR_DIM: -1})
            .drop_vars(config.YEAR_DIM)
        )
        innovations[{config.YEAR_DIM: 0}] = last_value
        forecasts_da = innovations.cumsum(dim=config.YEAR_DIM)
        overlap_first_index = 0 if self.overlap else 1
        forecasts_da = forecasts_da.isel({
            config.YEAR_DIM: slice(overlap_first_index, None)
        })

        if self.return_simulations:
            return forecasts_da

        bounds_da = bounds_from_simulations(
            forecasts_da, self.alpha, self.point_estimate
        )
        return bounds_da

    def _forecast_parameter_analytical(
            self,
            parameter_name: str,
            estimated_drift: xr.DataArray,
            estimated_std_of_errors: xr.DataArray,
            pred_years: np.ndarray,
        ) -> xr.DataArray:
        n_steps = xr.DataArray(
            np.arange(len(pred_years)),
            dims=[config.YEAR_DIM],
            coords={config.YEAR_DIM: pred_years},
        )

        last_value = (
            self.parameter_dataset[parameter_name]
            .isel({config.YEAR_DIM: -1})
            .drop_vars(config.YEAR_DIM)
        )

        forecasts_da = last_value + estimated_drift * n_steps
        forecasts_std = np.sqrt(n_steps) * estimated_std_of_errors

        overlap_first_index = 0 if self.overlap else 1
        year_slice = {config.YEAR_DIM: slice(overlap_first_index, None)}
        forecasts_da = forecasts_da.isel(year_slice)
        forecasts_std = forecasts_std.isel(year_slice)

        z_score = NormalDist().inv_cdf(1 - self.alpha / 2)
        z_multipliers = xr.DataArray(
            [-z_score, 0, z_score],
            dims=[config.BOUND_DIM],
            coords={config.BOUND_DIM: ["lower", "point", "upper"]},
        )

        combined_da = forecasts_da + z_multipliers * forecasts_std
        return combined_da

    