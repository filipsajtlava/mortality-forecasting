from typing import Literal

import xarray as xr
import numpy as np

from mortality_forecasting import config


class ForecastEvaluator:
    def __init__(
            self, 
            actual: xr.DataArray,
            forecast: xr.DataArray,
            aggregation: Literal["median", "mean"] = "median"
        ):
        self.actual = actual
        self.forecast = forecast
        self.aggregation = aggregation

    def _get_aggregates(self) -> xr.DataArray:
        if config.SIMULATION_DIM not in self.forecast.dims:
            return self.forecast

        if self.aggregation == "mean":
            agg_forecast = self.forecast.mean(dim=config.SIMULATION_DIM)
        elif self.aggregation == "median":
            agg_forecast = self.forecast.median(dim=config.SIMULATION_DIM)
        else:
            raise ValueError(f"Selected method '{self.aggregation}' isn't allowed.")
        return agg_forecast

    def mae(self) -> float:
        """Calculates the MAE of aggregated predictions and the test set

        Returns
        -------
            MAE error.
        """
        agg_forecast = self._get_aggregates()
        abs_errors = np.abs(self.actual - agg_forecast)
        return float(abs_errors.mean())

    def log_rmse(self) -> float:
        """Calculates the log-RMSE of aggregated predictions and the test set

        Returns
        -------
            RMSE error.
        """
        agg_forecast = self._get_aggregates()
        squared_errors = (np.log(agg_forecast) - np.log(self.actual)) ** 2
        return float(np.sqrt(squared_errors.mean()))    

    def mase(self, training: xr.DataArray) -> xr.DataArray:
        """Calculates the MASE of aggregated predictions and the test set
        for individual ages

        Returns
        -------
            MASE error.
        """
        agg_forecast = self._get_aggregates()
        abs_mean_errors = np.abs(
            self.actual - agg_forecast
        ).mean(dim=config.YEAR_DIM)

        training_diff_error = np.abs(
            training.diff(dim=config.YEAR_DIM)
        ).mean(dim=config.YEAR_DIM)
        return abs_mean_errors / training_diff_error
    
    def mser(self, training: xr.DataArray) -> xr.DataArray:
        """Calculates the MSEr of aggregated predictions and the test set
        for individual ages (MASE without the absolute value)

        Returns
        -------
            MSEr error.
        """
        agg_forecast = self._get_aggregates()
        mean_error_preds = (
            self.actual - agg_forecast
        ).mean(dim=config.YEAR_DIM)

        training_diff_error = np.abs(
            training.diff(dim=config.YEAR_DIM)
        ).mean(dim=config.YEAR_DIM)
        return mean_error_preds / training_diff_error