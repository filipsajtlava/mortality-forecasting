from typing import Self, Literal

import numpy as np
import xarray as xr

from data_downloading.datasets import MortalityDataset
from core.base_model import Model
from models.lee_carter import LeeCarterModel
from core.commons import validate_value_column, ParameterContainer
import config

class PoissonModel(Model):
    def __init__(
        self, 
        lee_miller_fix: bool = False,
        initialization: Literal["naive", "SVD"] = "naive",
        iterator_epsilon: float = 10e-9,
        verbose: bool = False
    ):
        self.iterator_epsilon = iterator_epsilon
        self.initialization = initialization
        self.verbose = verbose
        super().__init__(lee_miller_fix=lee_miller_fix)

    def fit(self, mortality_data: MortalityDataset, value_column: str) -> Self:
        validate_value_column(value_column)
        required_grids = ("E", "D") if self.initialization == "naive" else ("E", "D", "M")
        self._validate_dataset(
            mortality_data=mortality_data,
            required_grids=required_grids
        )

        self.mortality_data = mortality_data
        self.value_column = value_column
        self.D = self.mortality_data.D[self.value_column]
        self.E = self.mortality_data.E[self.value_column]

        ax, bx, kt = self._initialize_parameters()
        self.log_likelihood_history_ = [self._compute_log_likelihood(ax, bx, kt)]
        likelihood_change = np.inf
        iteration = 0

        while (
            likelihood_change > self.iterator_epsilon and 
            iteration < config.MAXIMUM_POISSON_ITERATIONS
        ):
            iteration += 1
            ax_new = self._get_new_alpha(ax, bx, kt)
            bx_new = self._get_new_beta(ax_new, bx, kt) # Each iteration takes new params
            kt_new = self._get_new_kappa(ax_new, bx_new, kt)
            self.log_likelihood_history_.append(
                self._compute_log_likelihood(ax_new, bx_new, kt_new)
            )
            likelihood_change = abs(
                self.log_likelihood_history_[-1] - 
                self.log_likelihood_history_[-2]
            )
            ax = ax_new
            bx = bx_new
            kt = kt_new

            if self.verbose:
                print(f"Iteration {iteration}: change in log-likelihood {likelihood_change}")

        if iteration >= config.MAXIMUM_POISSON_ITERATIONS:
            print(
                f"WARNING: the maximum amount of iterations " \
                f"({config.MAXIMUM_POISSON_ITERATIONS}) has been reached, " \
                f"so the algorithm might not have converged."
            )

        self.parameters_ = ParameterContainer(
            static=xr.Dataset(
                data_vars={
                    "ax": ax, 
                    "bx": bx
                }
            ),
            period=xr.Dataset(
                data_vars={
                    "kt": kt
                },                
                attrs={
                    "overlap": self.mortality_data.E.overlap,
                    "last_year": self.mortality_data.E.year_interval["end"]
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
        ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        ages = self.D[config.AGE_DIM].values
        years = self.D[config.YEAR_DIM].values

        if self.initialization == "SVD":
            lc_model = LeeCarterModel().fit(self.mortality_data, self.value_column)
            return (lc_model.ax_, lc_model.bx_, lc_model.kt_)
        elif self.initialization == "naive":
            ax = xr.DataArray(0, coords=[(config.AGE_DIM, ages)])
            bx = xr.DataArray(0, coords=[(config.AGE_DIM, ages)])
            kt = xr.DataArray(1, coords=[(config.YEAR_DIM, years)])
        else:
            raise ValueError("The selected initialization method is incorrect.")
        return (ax, bx, kt)

    def _get_new_alpha(
            self, 
            ax: xr.DataArray, 
            bx: xr.DataArray, 
            kt: xr.DataArray
        ) -> xr.DataArray:
        D_pred = self.E * np.exp(ax + bx * kt)
        ax_new = (
            ax + (self.D - D_pred).sum(dim=config.YEAR_DIM) / 
            D_pred.sum(dim=config.YEAR_DIM)
        )
        return ax_new
        
    def _get_new_beta(
            self, 
            ax: xr.DataArray, 
            bx: xr.DataArray, 
            kt: xr.DataArray
        ) -> xr.DataArray:
        D_pred = self.E * np.exp(ax + bx * kt)
        bx_new = (
            bx + (kt * (self.D - D_pred)).sum(dim=config.YEAR_DIM) / 
            (D_pred * kt*kt).sum(dim=config.YEAR_DIM)
        )
        bx_new = bx_new / bx_new.sum()
        return bx_new

    def _get_new_kappa(
            self, 
            ax: xr.DataArray, 
            bx: xr.DataArray, 
            kt: xr.DataArray
        ) -> xr.DataArray:
        D_pred = self.E * np.exp(ax + bx * kt)
        kt_new = (
            kt + (bx * (self.D - D_pred)).sum(dim=config.AGE_DIM) / 
            (D_pred * bx*bx).sum(dim=config.AGE_DIM)
        )
        kt_new = kt_new - kt_new.mean()
        return kt_new

    def _compute_log_likelihood(
            self, 
            ax: xr.DataArray, 
            bx: xr.DataArray, 
            kt: xr.DataArray
        ) -> float:
        log_likelihood = (
            self.D * (ax + bx * kt) - self.E * np.exp(ax + bx * kt)
        )
        return float(log_likelihood.sum())