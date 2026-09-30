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
            alpha: float = 0.05,
            return_simulations: bool = False,
            point_estimate: Literal["mean", "median"] = "median",
            **kwargs
        ) -> None:
        super().__init__(
            seed=seed,
            simulations=simulations,
            alpha=alpha,
            return_simulations=return_simulations,
            point_estimate=point_estimate,
            **kwargs
        )

    # TODO: keeping the outputs in dictionaries for now, more complex forecasters
    # may require special output containers, just like ParameterContainer, dont forget
    # to change this approach as well after that!
    def fit(self, parameter_dataset: xr.Dataset) -> None:
        self.parameter_dataset = parameter_dataset
        self.parameters_for_forecasting_ = {}

        for parameter_name, parameter_da in self.parameter_dataset.items():
            values = parameter_da.values
            drift_ = float(values[-1] - values[0]) / (len(values) - 1)
            std_of_errors_ = float(np.std(np.diff(values) - drift_, ddof=1))   
            self.parameters_for_forecasting_[parameter_name] = {
                "drift_": drift_,
                "std_of_errors_": std_of_errors_
            }

    def forecast_parameters(self, steps: int) -> xr.Dataset:
        forecasted_parameters = xr.Dataset()

        overlap_step = 0 if self.parameter_dataset.attrs["overlap"] else 1
        last_year = self.parameter_dataset.attrs["last_year"]
        pred_years = np.arange(overlap_step, steps + 1) + last_year

        for parameter_name, estimates in self.parameters_for_forecasting_.items():
            if self.simulations is not None:
                forecasts_da = self._forecast_parameter_stochastic(
                    parameter_name,
                    estimates["drift_"],
                    estimates["std_of_errors_"],
                    steps,
                    overlap_step,
                    pred_years,
                )
            else:
                forecasts_da = self._forecast_parameter_analytical(
                    parameter_name,
                    estimates["drift_"],
                    estimates["std_of_errors_"],
                    steps,
                    overlap_step,
                    pred_years
                )
            forecasted_parameters[parameter_name] = forecasts_da
        return forecasted_parameters

    def _forecast_parameter_stochastic(
            self, 
            parameter_name: str,
            estimated_drift: float,
            estimated_std_of_errors: float,
            steps: int,
            overlap_step: int,
            pred_years: np.ndarray,
        ) -> xr.DataArray:
        innovations = self.seed.normal(
            estimated_drift,
            estimated_std_of_errors,
            size=(steps, self.simulations)
        )
        innovations = np.insert(innovations, 0, 0, axis=0)    
        forecasted_values = (
            self.parameter_dataset[parameter_name].values[-1] + 
            np.cumsum(innovations, axis=0)
        )
        forecasts_da = xr.DataArray(
            forecasted_values[overlap_step:],
            coords=[pred_years, np.arange(1, self.simulations + 1)],
            dims=[config.YEAR_DIM, config.SIMULATION_DIM]
        )

        if self.return_simulations:
            return forecasts_da

        bounds_da = bounds_from_simulations(
            forecasts_da, self.alpha, self.point_estimate
        )
        return bounds_da

    def _forecast_parameter_analytical(
            self, 
            parameter_name: str,
            estimated_drift: float,
            estimated_std_of_errors: float,
            steps: int,
            overlap_step: int,
            pred_years: np.ndarray,     
        ) -> xr.DataArray:
        drifts = np.repeat(estimated_drift, steps)
        drifts = np.insert(drifts, 0, 0)
        forecasted_values = (
            self.parameter_dataset[parameter_name].values[-1] +
            np.cumsum(drifts, axis=0)
        )
        point_forecasts_da = xr.DataArray(
            forecasted_values[overlap_step:],
            coords=[pred_years],
            dims=[config.YEAR_DIM]
        )

        z_score = NormalDist().inv_cdf(1 - self.alpha / 2)
        z_multipliers = xr.DataArray(
            [-z_score, 0, z_score],
            dims=[config.BOUND_DIM],
            coords={config.BOUND_DIM: ["lower", "point", "upper"]}
        )
        forecasts_std = xr.DataArray(
            np.sqrt(np.arange(overlap_step, steps + 1)) * estimated_std_of_errors,
            coords={config.YEAR_DIM: pred_years},
            dims=[config.YEAR_DIM]
        )

        combined_da = point_forecasts_da + z_multipliers * forecasts_std
        return combined_da

    