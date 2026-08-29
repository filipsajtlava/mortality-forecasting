from typing import Literal

import xarray as xr
import numpy as np

from mortality_forecasting import config


class ForecastEvaluator:
    def __init__(
            self, 
            actual: xr.DataArray,
            forecast: xr.DataArray,
            point_estimate: Literal["median", "mean"] = "median"
        ):
        self.actual = actual
        self.forecast = forecast
        self.point_estimate = point_estimate

    def _get_aggregates(self) -> np.ndarray:
        if config.BOUND_DIM in self.forecast.dims:
            clean_data = self.forecast.sel({config.BOUND_DIM: "point"})
        elif config.SIMULATION_DIM in self.forecast.dims:
            clean_data = getattr(self.forecast, self.point_estimate)(
                dim=config.SIMULATION_DIM
            )
        else:
            clean_data = self.forecast
        return clean_data.to_numpy()

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