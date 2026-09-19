from typing import Literal, Self

import xarray as xr
import numpy as np

from mortality_forecasting import config
from mortality_forecasting.core._commons import ParameterContainer, validate_value_column
from mortality_forecasting.core._base_model import Model
from mortality_forecasting.core._base_glm import GLMCapable
from mortality_forecasting.models._lee_carter import LeeCarterModel
from mortality_forecasting.data_processing._dataset import MortalityDataset


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

    def fit(self, mortality_data: MortalityDataset, value_column: str) -> Self:
        validate_value_column(value_column)
        self._validate_dataset(
            mortality_data=mortality_data,
            value_column=value_column
        )

        self.mortality_data = mortality_data
        self.value_column = value_column
        self.D = self.mortality_data.D[self.value_column]
        self.E = self.mortality_data.E[self.value_column]

        ax, bx, kt, lambda_dispersion = self._initialize_parameters()
        self.log_likelihood_history_ = [
            self._compute_log_likelihood(ax, bx, kt, lambda_dispersion)
        ]
        likelihood_change = np.inf
        iteration = 0

        while (
            likelihood_change > self.ftol and 
            iteration <= config.MAXIMUM_POISSON_ITERATIONS
        ):
            iteration += 1
            ax_new = self._get_new_alpha(ax, bx, kt, lambda_dispersion)
            bx_new = self._get_new_beta(ax_new, bx, kt, lambda_dispersion)
            kt_new = self._get_new_kappa(ax_new, bx_new, kt, lambda_dispersion)
            lambda_dispersion_new = self._get_new_lambda(
                ax_new, 
                bx_new, 
                kt_new, 
                lambda_dispersion
            )
            self.log_likelihood_history_.append(
                self._compute_log_likelihood(ax_new, bx_new, kt_new, lambda_dispersion_new)
            )
            likelihood_change = abs(
                (self.log_likelihood_history_[-1] - self.log_likelihood_history_[-2]) /
                self.log_likelihood_history_[-2]
            )
            ax = ax_new
            bx = bx_new
            kt = kt_new
            lambda_dispersion = lambda_dispersion_new

            if self.verbose:
                print(
                    f"Iteration {iteration} - relative change " \
                    f"in log-likelihood: {likelihood_change}"
                )

        if iteration > config.MAXIMUM_POISSON_ITERATIONS:
            print(
                f"WARNING: the maximum amount of iterations " \
                f"({config.MAXIMUM_POISSON_ITERATIONS}) has been reached, " \
                f"so the algorithm might not have converged."
            )

        if self.lee_miller_fix:
            last_year = self.mortality_data.D.year_interval["end"]
            M_last_column = (self.D / self.E).sel({config.YEAR_DIM: last_year})
            ax = (
                np.log(M_last_column) - bx * kt.sel({config.YEAR_DIM: last_year})
            ).drop_vars(config.YEAR_DIM)

        self.parameters_ = ParameterContainer(
            static=xr.Dataset(
                data_vars={
                    "ax": ax, 
                    "bx": bx,
                    "lambda": lambda_dispersion
                }
            ),
            period=xr.Dataset(
                data_vars={
                    "kt": kt
                },                
                attrs={
                    "overlap": self.mortality_data.E.overlap,
                    "last_year": self.mortality_data.E.year_interval["end"],
                }
            )
        )
        return self

    def _predict_mortalities(
            self, 
            forecasted_values: ParameterContainer
        ) -> xr.DataArray:
        log_M_predictions = (
            forecasted_values.static.ax + 
            forecasted_values.static.bx * forecasted_values.period.kt
        )
        return np.exp(log_M_predictions)

    def _initialize_parameters(
            self
        ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, int]:
        ages = self.D[config.AGE_DIM].values
        years = self.D[config.YEAR_DIM].values

        lambda_dispersion = 1
        if self.initialization == "SVD":
            lc_model = LeeCarterModel().fit(self.mortality_data, self.value_column)
            return (
                lc_model.parameters_["ax"],
                lc_model.parameters_["bx"],
                lc_model.parameters_["kt"],
                lambda_dispersion
            )
        elif self.initialization == "naive":
            ax = xr.DataArray(0, coords=[(config.AGE_DIM, ages)])
            bx = xr.DataArray(0, coords=[(config.AGE_DIM, ages)])
            kt = xr.DataArray(1, coords=[(config.YEAR_DIM, years)])
        else:
            raise ValueError("The selected initialization method is incorrect.")
        return (ax, bx, kt, lambda_dispersion)

    def _get_new_alpha(
            self, 
            ax: xr.DataArray, 
            bx: xr.DataArray, 
            kt: xr.DataArray,
            lambda_dispersion: float
        ) -> xr.DataArray:
        D_pred = self.E * np.exp(ax + bx * kt)
        D_pred_score = D_pred * (lambda_dispersion + self.D) / (lambda_dispersion + D_pred)
        D_pred_hessian = D_pred * (
            lambda_dispersion * (lambda_dispersion + self.D) / 
            (lambda_dispersion + D_pred) ** 2
        )

        ax_new = (
            ax + (self.D - D_pred_score).sum(dim=config.YEAR_DIM) / 
            D_pred_hessian.sum(dim=config.YEAR_DIM)
        )
        return ax_new
        
    def _get_new_beta(
            self, 
            ax: xr.DataArray, 
            bx: xr.DataArray, 
            kt: xr.DataArray,
            lambda_dispersion: float
        ) -> xr.DataArray:
        D_pred = self.E * np.exp(ax + bx * kt)
        D_pred_score = D_pred * (lambda_dispersion + self.D) / (lambda_dispersion + D_pred)
        D_pred_hessian = D_pred * (
            lambda_dispersion * (lambda_dispersion + self.D) / 
            (lambda_dispersion + D_pred) ** 2
        )

        bx_new = (
            bx + (kt * (self.D - D_pred_score)).sum(dim=config.YEAR_DIM) / 
            (D_pred_hessian * kt*kt).sum(dim=config.YEAR_DIM)
        )
        bx_new = bx_new / bx_new.sum()
        return bx_new

    def _get_new_kappa(
            self, 
            ax: xr.DataArray, 
            bx: xr.DataArray, 
            kt: xr.DataArray,
            lambda_dispersion: float
        ) -> xr.DataArray:
        D_pred = self.E * np.exp(ax + bx * kt)
        D_pred_score = D_pred * (lambda_dispersion + self.D) / (lambda_dispersion + D_pred)
        D_pred_hessian = D_pred * (
            lambda_dispersion * (lambda_dispersion + self.D) / 
            (lambda_dispersion + D_pred) ** 2
        )

        kt_new = (
            kt + (bx * (self.D - D_pred_score)).sum(dim=config.AGE_DIM) / 
            (D_pred_hessian * bx*bx).sum(dim=config.AGE_DIM)
        )
        kt_new = kt_new - kt_new.mean()
        return kt_new

    def _get_new_lambda(
            self,
            ax: xr.DataArray, 
            bx: xr.DataArray, 
            kt: xr.DataArray,
            lambda_dispersion: float
        ) -> float:
        D_pred = self.E * np.exp(ax + bx * kt)
        max_death = self.D.max()
        D_sum_numerator = np.insert(
            np.cumsum(1 / (lambda_dispersion + np.arange(0, max_death))), 0, 0
        )
        D_sum_denominator = np.insert(
            np.cumsum(1 / (lambda_dispersion + np.arange(0, max_death)) ** 2), 0, 0
        )

        lambda_dispersion_new = (
            lambda_dispersion + (
                D_sum_numerator[self.D.astype(int)] + 
                np.log(lambda_dispersion / (lambda_dispersion + D_pred)) + 
                (D_pred - self.D) / (D_pred + lambda_dispersion)
            ).sum() / (
                D_sum_denominator[self.D.astype(int)] - 
                1 / lambda_dispersion +
                (2 * D_pred + lambda_dispersion - self.D) / (lambda_dispersion + D_pred) ** 2 
            ).sum()
        )
        return float(lambda_dispersion_new)
        
    def _compute_log_likelihood(
            self, 
            ax: xr.DataArray, 
            bx: xr.DataArray, 
            kt: xr.DataArray,
            lambda_dispersion: float
        ) -> float:
        D_pred = self.E * np.exp(ax + bx * kt)
        max_death = self.D.max()
        D_sum_lookup = np.insert(
            np.cumsum(np.log(lambda_dispersion + np.arange(0, max_death))), 0, 0
        )

        log_likelihood = (
            D_sum_lookup[self.D.astype(int)] - (self.D + lambda_dispersion) * 
            np.log(1 + D_pred / lambda_dispersion) +
            self.D * np.log(D_pred / lambda_dispersion)
        )
        return float(log_likelihood.sum())