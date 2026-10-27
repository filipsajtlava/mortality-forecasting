import xarray as xr

from mortality_forecasting.core._base_forecaster import Forecaster
from mortality_forecasting.core._commons import ParameterContainer


class DualForecaster:
    def __init__(self, period: Forecaster, cohort: Forecaster) -> None:
        self.period = period
        self.cohort = cohort

    def fit(self, parameter_container: ParameterContainer) -> None:
        if parameter_container.cohort is None:
            raise ValueError(
                "This model is incompatible with a dual forecaster, " \
                "as it does not contain a cohort component."
            )

        self.period.fit(parameter_container.period)
        self.cohort.fit(parameter_container.cohort)

    def forecast_parameters(
            self, 
            steps: int,
            alpha: float
        ) -> tuple[xr.Dataset, xr.Dataset]:
        period_forecasted = self.period.forecast_parameters(steps, alpha)
        cohort_forecasted = self.cohort.forecast_parameters(steps, alpha)
        return period_forecasted, cohort_forecasted