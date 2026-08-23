import numpy as np
import xarray as xr

from core.base_forecaster import Forecaster
import config

class RandomWalkWithDrift(Forecaster):
    def __init__(
            self, 
            seed: int | np.random.Generator | None = None,
            stochastic: bool = True
        ) -> None:
        super().__init__(seed=seed)
        self.stochastic = stochastic

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

    def forecast_parameters(self, steps: int, simulations: int) -> xr.Dataset:
        rng = self._normalize_seed()
        output_dataset = xr.Dataset()

        overlap_step = 0 if self.parameter_dataset.attrs["overlap"] else 1
        last_year = self.parameter_dataset.attrs["last_year"]
        pred_years = np.arange(overlap_step, steps + 1) + last_year

        for parameter_name, estimates in self.parameters_for_forecasting_.items():
            if self.stochastic:
                forecasted_da = self._forecast_parameter_stochastic(
                    parameter_name,
                    estimates["drift_"],
                    estimates["std_of_errors_"],
                    steps,
                    simulations,
                    overlap_step,
                    pred_years,
                    rng
                )
            else:
                forecasted_da = self._forecast_parameter_analytical(
                    parameter_name,
                    estimates["drift_"],
                    steps,
                    overlap_step,
                    pred_years
                )
            output_dataset[parameter_name] = forecasted_da
        return output_dataset

    def _forecast_parameter_stochastic(
            self, 
            parameter_name: str,
            estimated_drift: float,
            estimated_std_of_errors: float,
            steps: int,
            simulations: int,
            overlap_step: int,
            pred_years: np.ndarray,
            rng: np.random.Generator
        ) -> xr.DataArray:
        innovations = rng.normal(
            estimated_drift,
            estimated_std_of_errors,
            size=(steps, simulations)
        )
        innovations = np.insert(innovations, 0, 0, axis=0)    
        forecasted_values = (
            self.parameter_dataset[parameter_name].values[-1] + 
            np.cumsum(innovations, axis=0)
        )

        return xr.DataArray(
            forecasted_values[overlap_step:],
            coords=[pred_years, np.arange(1, simulations + 1)],
            dims=[config.YEAR_DIM, config.SIMULATION_DIM]
        )

    def _forecast_parameter_analytical(
            self, 
            parameter_name: str,
            estimated_drift: float,
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

        return xr.DataArray(
            forecasted_values[overlap_step:],
            coords=[pred_years],
            dims=[config.YEAR_DIM]
        )

    