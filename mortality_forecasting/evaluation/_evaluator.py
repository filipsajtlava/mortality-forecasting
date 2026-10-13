import xarray as xr
import numpy as np

from mortality_forecasting import config
from mortality_forecasting.plotting._evaluator_plot import EvaluatorPlotter
from mortality_forecasting.core._commons import ForecastContainer


class ForecastEvaluator:
    def __init__(
            self, 
            actual: xr.DataArray,
            forecast_container: ForecastContainer
        ) -> None:
        self.actual = actual
        self.forecast_container = forecast_container
        self.point_estimate = self._normalize_point_estimate()

    def _normalize_point_estimate(self) -> xr.DataArray:
        return self.forecast_container.mortality_rates_.sel(
            {config.BOUND_DIM: "point"}
        ).drop_vars(config.BOUND_DIM)

    @property
    def plot(self) -> EvaluatorPlotter:
        return EvaluatorPlotter(self)

    def mae(self) -> float:
        """Calculates MAE of aggregated predictions and test set."""
        return float(np.abs(self.actual - self.point_estimate).mean())

    def log_rmse(self) -> float:
        """Calculates log-RMSE of predictions and test set."""
        squared_errors = (
            np.log(self.point_estimate) - np.log(self.actual)
        ) ** 2
        return float(np.sqrt(squared_errors.mean()))    

    def mase(self, training_data: xr.DataArray) -> xr.DataArray:
        """Calculates MASE for individual ages."""
        abs_mean_errors = np.abs(
            self.actual - self.point_estimate
        ).mean(dim=config.YEAR_DIM)
        training_diff_error = np.abs(
            training_data.diff(dim=config.YEAR_DIM)
        ).mean(dim=config.YEAR_DIM)
        return abs_mean_errors / training_diff_error
    
    def mser(self, training_data: xr.DataArray) -> xr.DataArray:
        """Calculates MSEr (MASE without absolute value) for ages."""
        mean_error_preds = (
            self.actual - self.point_estimate
        ).mean(dim=config.YEAR_DIM)
        training_diff_error = np.abs(
            training_data.diff(dim=config.YEAR_DIM)
        ).mean(dim=config.YEAR_DIM)
        return mean_error_preds / training_diff_error

    def rcs(self) -> xr.DataArray:
        upper = self.forecast_container.mortality_rates_.sel(
            {config.BOUND_DIM: "upper"}
        )
        lower = self.forecast_container.mortality_rates_.sel(
            {config.BOUND_DIM: "lower"}
        )
        return (
            (self.actual <= upper) & (self.actual >= lower)
        ).sum(dim=config.YEAR_DIM) / self.actual.sizes[config.YEAR_DIM]

    def mws(self) -> xr.DataArray:
        upper = self.forecast_container.mortality_rates_.sel(
            {config.BOUND_DIM: "upper"}
        ).drop_vars(config.BOUND_DIM)
        lower = self.forecast_container.mortality_rates_.sel(
            {config.BOUND_DIM: "lower"}
        ).drop_vars(config.BOUND_DIM)
        alpha = self.forecast_container.attrs["alpha"]

        diff1 = lower - self.actual
        diff2 = self.actual - upper
        zeros = xr.zeros_like(diff1)
        penalty = xr.concat(
            [diff1, diff2, zeros], dim="arrays"
        ).max(dim="arrays")

        interval_width = (
            upper - lower
        ).sum(dim=config.YEAR_DIM)
        return interval_width + (2 / alpha) * penalty.sum(
            dim=config.YEAR_DIM
        )