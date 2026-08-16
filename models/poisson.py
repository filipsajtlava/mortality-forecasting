from typing import Self, Literal

import numpy as np
import xarray as xr

from data_downloading.datasets import MortalityDataset
from core.base_model import Model
from models.lee_carter import LeeCarterModel
from core.commons import validate_value_column
from config import MAXIMUM_POISSON_ITERATIONS

class PoissonModel(Model):
    _PARAM_CHOICES = {
        **Model._PARAM_CHOICES,
        "initialization": ("naive", "SVD")
    }

    def __init__(
        self, 
        lee_miller_fix: bool = False,
        seed: int | np.random.Generator | None = None,
        initialization: Literal["naive", "SVD"] = "naive",
        iterator_epsilon: float = 10e-9
    ):
        self.iterator_epsilon = iterator_epsilon
        self.initialization = initialization
        super().__init__(lee_miller_fix=lee_miller_fix, seed=seed)

    def fit(self, mortality_data: MortalityDataset, value_column: str) -> Self:
        required_grids = ("E", "D") if self.initialization == "naive" else ("E", "D", "M")
        self._validate_hyperparameters()
        validate_value_column(value_column)
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

        while likelihood_change > self.iterator_epsilon and iteration < MAXIMUM_POISSON_ITERATIONS:
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
            print(f"Iteration: {iteration}: {float(likelihood_change)}")

        if iteration >= MAXIMUM_POISSON_ITERATIONS:
            print(
                f"WARNING: the maximum amount of iterations " \
                f"({MAXIMUM_POISSON_ITERATIONS}) has been reached, " \
                f"so the algorithm might not have converged."
            )

        self.ax_ = ax
        self.bx_ = bx
        self.kt_ = kt
        return self

    def _initialize_parameters(
            self
        ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        ages = self.D.Age.values
        years = self.D.Year.values

        if self.initialization == "SVD":
            lc_model = LeeCarterModel().fit(self.mortality_data, self.value_column)
            return (lc_model.ax_, lc_model.bx_, lc_model.kt_)
        elif self.initialization == "naive":
            ax = xr.DataArray(0, coords=[("Age", ages)])
            bx = xr.DataArray(0, coords=[("Age", ages)])
            kt = xr.DataArray(1, coords=[("Year", years)])

        return (ax, bx, kt)

    def _calculate_log_mortality(
            self, 
            ax: xr.DataArray, 
            bx: xr.DataArray,
            kt: xr.DataArray
        ) -> xr.DataArray:
        return ax + bx * kt

    def _get_new_alpha(
            self, 
            ax: xr.DataArray, 
            bx: xr.DataArray, 
            kt: xr.DataArray
        ) -> xr.DataArray:
        D_pred = self.E * np.exp(ax + bx * kt)
        ax_new = (
            ax + (self.D - D_pred).sum(dim="Year") / D_pred.sum(dim="Year")
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
            bx + (kt * (self.D - D_pred)).sum(dim="Year") / 
            (D_pred * kt*kt).sum(dim="Year")
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
            kt + (bx * (self.D - D_pred)).sum(dim="Age") / 
            (D_pred * bx*bx).sum(dim="Age")
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

    def predict(self, steps: int, simulations: int = 1) -> xr.DataArray:
        return steps