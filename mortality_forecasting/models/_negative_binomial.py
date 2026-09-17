from typing import Literal

import xarray as xr
import numpy as np

from mortality_forecasting import config
from mortality_forecasting.core._commons import ParameterContainer
from mortality_forecasting.core._base_model import Model
from mortality_forecasting.core._base_glm import GLMCapable
from mortality_forecasting.models._lee_carter import LeeCarterModel


class NegativeBinomialModel(Model, GLMCapable):
    def __init__(
            self, 
            lee_miller_fix: bool = False,
            initialization: Literal["naive", "SVD"] = "SVD",
            ftol: float = 1e-5,
            verbose: bool = False
        ):
        self.initialization = initialization
        self.ftol = ftol
        self.verbose = verbose
        super().__init__(lee_miller_fix=lee_miller_fix)

    def fit(self):
        pass

    def _predict_mortalities(
            self, 
            forecasted_values: ParameterContainer
        ) -> xr.DataArray:
        log_M_predictions = (
            forecasted_values.static.ax + 
            forecasted_values.static.bx * forecasted_values.period.kt
        )
        return log_M_predictions

    def _initialize_parameters(
            self
        ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, int]:
        ages = self.D[config.AGE_DIM].values
        years = self.D[config.YEAR_DIM].values

        lambda_param = 1
        if self.initialization == "SVD":
            lc_model = LeeCarterModel().fit(self.mortality_data, self.value_column)
            return (
                lc_model.parameters_["ax"],
                lc_model.parameters_["bx"],
                lc_model.parameters_["kt"],
                lambda_param
            )
        elif self.initialization == "naive":
            ax = xr.DataArray(0, coords=[(config.AGE_DIM, ages)])
            bx = xr.DataArray(0, coords=[(config.AGE_DIM, ages)])
            kt = xr.DataArray(1, coords=[(config.YEAR_DIM, years)])
        else:
            raise ValueError("The selected initialization method is incorrect.")
        return (ax, bx, kt, lambda_param)

    def _compute_log_likelihood(self):
        pass