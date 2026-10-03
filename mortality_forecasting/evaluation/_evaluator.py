from typing import Literal

import xarray as xr
import numpy as np

from mortality_forecasting import config
from mortality_forecasting.plotting._evaluator_plot import EvaluatorPlotter
from mortality_forecasting.core._commons import ForecastContainer


class ForecastEvaluator:
    def __init__(
            self, 
            actual: xr.DataArray,
            forecast: ForecastContainer
        ) -> None:
        self.actual = actual
        self.forecast = forecast
        self.point_estimate = self._normalize_point_estimate()

    def _normalize_point_estimate(self) -> np.ndarray:
        point_da = self.forecast.mortality_rates_.sel({config.BOUND_DIM: "point"})
        return point_da.to_numpy()

    @property
    def plot(self) -> EvaluatorPlotter:
        return EvaluatorPlotter(self)

    def mae(self) -> float:
        """Calculates the MAE of aggregated predictions and the test set

        Returns
        -------
            MAE error.
        """
        abs_errors = np.abs(self.actual - self.point_estimate)
        return float(abs_errors.mean())

    def log_rmse(self) -> float:
        """Calculates the log-RMSE of aggregated predictions and the test set

        Returns
        -------
            RMSE error.
        """
        squared_errors = (np.log(self.point_estimate) - np.log(self.actual)) ** 2
        return float(np.sqrt(squared_errors.mean()))    

    def mase(self, training_data: xr.DataArray) -> xr.DataArray:
        """Calculates the MASE of aggregated predictions and the test set
        for individual ages

        Returns
        -------
            MASE error.
        """
        abs_mean_errors = np.abs(
            self.actual - self.point_estimate
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
            self.actual - self.point_estimate
        ).mean(dim=config.YEAR_DIM)

        training_diff_error = np.abs(
            training_data.diff(dim=config.YEAR_DIM)
        ).mean(dim=config.YEAR_DIM)
        return mean_error_preds / training_diff_error

    def rcs(self) -> xr.DataArray:
        upper_quantiles = self.forecast.mortality_rates_.sel({config.BOUND_DIM: "upper"})
        lower_quantiles = self.forecast.mortality_rates_.sel({config.BOUND_DIM: "lower"})
        rcs = (
            (self.actual <= upper_quantiles) & (self.actual >= lower_quantiles)
        ).sum(dim=config.YEAR_DIM) / (
            self.actual.sizes[config.YEAR_DIM]
        )
        return rcs

    def mws(self):
        upper_quantiles = self.forecast.mortality_rates_.sel({config.BOUND_DIM: "upper"}).drop_vars(config.BOUND_DIM)
        lower_quantiles = self.forecast.mortality_rates_.sel({config.BOUND_DIM: "lower"}).drop_vars(config.BOUND_DIM)
        alpha = self.forecast.attrs["alpha"]

        diff1 = lower_quantiles - self.actual
        diff2 = self.actual - upper_quantiles
        zeros = xr.zeros_like(diff1)
        penalty = xr.concat([diff1, diff2, zeros], dim="arrays").max(dim="arrays")

        interval_width = (upper_quantiles - lower_quantiles).sum(dim=config.YEAR_DIM)
        mws = interval_width + (2 / alpha) * penalty.sum(dim=config.YEAR_DIM)
        return mws