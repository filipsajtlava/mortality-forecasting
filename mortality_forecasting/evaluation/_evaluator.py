from typing import Literal

import xarray as xr
import numpy as np

from mortality_forecasting import config
from mortality_forecasting.core._commons import bounds_from_simulations
from mortality_forecasting.plotting._evaluator_plot import EvaluatorPlotter

# TODO: enfore the forecast to be the entire ForecastContainer or the .mortalities_
# but dont allow the user to pass only the single bound="point"
class ForecastEvaluator:
    def __init__(
            self, 
            actual: xr.DataArray,
            forecast: xr.DataArray,
            alpha: float = 0.05,
            point_estimate: Literal["median", "mean"] = "median"
        ) -> None:
        self.actual = actual
        self.alpha = alpha
        self.point_estimate = point_estimate
        self.forecast = self._aggregate_forecasts(forecast)

    def _aggregate_forecasts(self, forecast: xr.DataArray) -> np.ndarray:
        forecast = bounds_from_simulations(forecast, self.alpha, self.point_estimate)
        return forecast

    @property
    def plot(self) -> EvaluatorPlotter:
        return EvaluatorPlotter(self)

    def mae(self) -> float:
        """Calculates the MAE of aggregated predictions and the test set

        Returns
        -------
            MAE error.
        """
        abs_errors = np.abs(self.actual - self.forecast.sel({config.BOUND_DIM: "point"}))
        return float(abs_errors.mean())

    def log_rmse(self) -> float:
        """Calculates the log-RMSE of aggregated predictions and the test set

        Returns
        -------
            RMSE error.
        """
        squared_errors = (
            np.log(self.forecast.sel({config.BOUND_DIM: "point"})) - 
            np.log(self.actual)
        ) ** 2
        return float(np.sqrt(squared_errors.mean()))    

    def mase(self, training_data: xr.DataArray) -> xr.DataArray:
        """Calculates the MASE of aggregated predictions and the test set
        for individual ages

        Returns
        -------
            MASE error.
        """
        abs_mean_errors = np.abs(
            self.actual - self.forecast.sel({config.BOUND_DIM: "point"})
        ).mean(dim=config.YEAR_DIM)

        training_diff_error = np.abs(
            training_data.diff(dim=config.YEAR_DIM)
        ).mean(dim=config.YEAR_DIM)
        return abs_mean_errors / training_diff_error
    
    def mser(self, training_data: xr.DataArray) -> xr.DataArray:
        """Calculates the MSEr of aggregated predictions and the test set
        for individual ages (MASE without the absolute value)

        Returns
        -------
            MSEr error.
        """
        mean_error_preds = (
            self.actual - self.forecast.sel({config.BOUND_DIM: "point"})
        ).mean(dim=config.YEAR_DIM)

        training_diff_error = np.abs(
            training_data.diff(dim=config.YEAR_DIM)
        ).mean(dim=config.YEAR_DIM)
        return mean_error_preds / training_diff_error