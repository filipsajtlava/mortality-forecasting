import xarray as xr
import numpy as np

from mortality_forecasting import config
from mortality_forecasting.plotting._evaluator_plot import EvaluatorPlotter
from mortality_forecasting.core._commons import ForecastContainer


class ForecastEvaluator:
    def __init__(
            self, 
            actual: xr.DataArray | xr.Dataset,
            forecast_container: ForecastContainer
        ) -> None:
        self.actual = self._normalize_dataset_input(actual)
        self.forecast_container = forecast_container
        self._bound_dims = self.forecast_container.attrs["bound_dims"]
        self.point_estimate = self._normalize_point_estimate()

        self.actual = self._check_year_validity()

    def _normalize_dataset_input(
            self, 
            actual: xr.DataArray | xr.Dataset,
            input_matrix: str = "M"
        ) -> xr.DataArray:
        if isinstance(actual, xr.Dataset):
            if input_matrix in actual:
                return actual[input_matrix]
            else:
                raise ValueError("Please provide a dataset that contains the mortality matrix 'M'")
        return actual

    def _normalize_point_estimate(self) -> xr.DataArray:
        return self.forecast_container.mortality_rates_.sel(
            {config.BOUND_DIM: "point"}
        ).drop_vars(config.BOUND_DIM)

    def _check_year_validity(self) -> xr.DataArray:
        actual_years = self.actual.coords[config.YEAR_DIM].values
        forecast_years = self.point_estimate.coords[config.YEAR_DIM].values

        if np.array_equal(actual_years, forecast_years):
            return self.actual
        else:
            if set(forecast_years).issubset(set(actual_years)):
                print("WARNING: only the years included in the forecast will be used for evaluation.")
                return self.actual.sel({config.YEAR_DIM: forecast_years})
            else:
                raise ValueError("Non-itersecting year spans are incompatible.")

    @property
    def plot(self) -> EvaluatorPlotter:
        return EvaluatorPlotter(self)

    def mae(self) -> float:
        """Calculates MAE of aggregated predictions and test set."""
        return np.abs(self.actual - self.point_estimate).mean(dim=[config.AGE_DIM, config.YEAR_DIM])

    def log_rmse(self) -> float:
        """Calculates log-RMSE of predictions and test set."""
        squared_errors = (
            np.log(self.point_estimate) - np.log(self.actual)
        ) ** 2
        return np.sqrt(squared_errors.mean(dim=[config.AGE_DIM, config.YEAR_DIM]))   

    def mase(
            self, 
            training_data: xr.DataArray | xr.Dataset
        ) -> xr.DataArray:
        """Calculates MASE for individual ages."""
        training_data = self._normalize_dataset_input(training_data)
        abs_mean_errors = np.abs(
            self.actual - self.point_estimate
        ).mean(dim=config.YEAR_DIM)
        training_diff_error = np.abs(
            training_data.diff(dim=config.YEAR_DIM)
        ).mean(dim=config.YEAR_DIM)
        return abs_mean_errors / training_diff_error
    
    def mser(
            self, 
            training_data: xr.DataArray | xr.Dataset
        ) -> xr.DataArray:
        """Calculates MSEr (MASE without absolute value) for ages."""
        training_data = self._normalize_dataset_input(training_data)
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