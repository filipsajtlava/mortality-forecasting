from typing import Self

import numpy as np
import xarray as xr
import pandas as pd

from data_downloading.datasets import MortalityDataset
from core.base_model import Model
from core.commons import validate_value_column
from config import MAXIMUM_POISSON_ITERATIONS

class PoissonModel(Model):
    _REQUIRED_GRIDS = ("E", "D")
    _PARAM_CHOICES = {
        **Model._PARAM_CHOICES,
        "initialization": ("naive", "SVD")
    }

    def __init__(
        self, 
        lee_miller_fix: bool = False,
        seed: int | np.random.Generator | None = None,
        initialization: str = "naive",
        iterator_epsilon: float = 0.01
    ):
        self.iterator_epsilon = iterator_epsilon
        self.initialization = initialization # TODO: implement into initialize params
        super().__init__(lee_miller_fix=lee_miller_fix, seed=seed)

    def fit(self, mortality_data: MortalityDataset, value_column: str) -> Self:
        self._validate_hyperparameters()
        validate_value_column(value_column)
        self._validate_dataset(
            mortality_data=mortality_data,
            required_grids=self._REQUIRED_GRIDS
        )

        self.mortality_data = mortality_data
        self.value_column = value_column
        self.D = self.mortality_data.D[self.value_column]
        self.E = self.mortality_data.E[self.value_column]

        self._initialize_parameters()
        log_likelihood_previous = 0
        likelihood_change = np.inf
        iteration = 0

        while likelihood_change > self.iterator_epsilon and iteration < MAXIMUM_POISSON_ITERATIONS:
            iteration += 1
            self._alpha_update(iteration)
            self._beta_update(iteration)
            self._kappa_update(iteration)

            log_likelihood = self._compute_log_likelihood(iteration)
            likelihood_change = abs(log_likelihood - log_likelihood_previous)
            log_likelihood_previous = log_likelihood
            #print(f"Iteration: {iteration}: {float(likelihood_change)}")

        self.ax_history_ = self.ax_history_.sel(Iteration=slice(0, iteration))
        self.bx_history_ = self.bx_history_.sel(Iteration=slice(0, iteration))
        self.kt_history_ = self.kt_history_.sel(Iteration=slice(0, iteration))
        return self

    def _initialize_parameters(self):
        ages = self.D.Age
        years = self.D.Year
        iterations = np.arange(0, MAXIMUM_POISSON_ITERATIONS + 1)

        self.ax_history_ = xr.DataArray(
            np.full([len(ages), len(iterations)], np.nan),
            dims=["Age", "Iteration"],
            coords={
                "Age": ages,
                "Iteration": iterations
            }
        )
        self.bx_history_ = xr.DataArray(
            np.full([len(ages), len(iterations)], np.nan),
            dims=["Age", "Iteration"],
            coords={
                "Age": ages,
                "Iteration": iterations
            }
        )
        self.kt_history_ = xr.DataArray(
            np.full([len(years), len(iterations)], np.nan),
            dims=["Year", "Iteration"],
            coords={
                "Year": years,
                "Iteration": iterations
            }
        )

        if self.initialization == "naive":
            self.ax_history_.loc[{"Iteration": 0}] = 0
            self.bx_history_.loc[{"Iteration": 0}] = 0
            self.kt_history_.loc[{"Iteration": 0}] = 1

    def _compute_log_likelihood(self, iteration: int) -> float:
        log_likelihood = self.D * (
            self.ax_history_.sel(Iteration=iteration) +
            self.bx_history_.sel(Iteration=iteration) *
            self.kt_history_.sel(Iteration=iteration)
        ) - self.E * np.exp(
            self.ax_history_.sel(Iteration=iteration) +
            self.bx_history_.sel(Iteration=iteration) *
            self.kt_history_.sel(Iteration=iteration)
        )
        return log_likelihood.sum()

    def _alpha_update(self, iteration: int) -> None:
        D_pred = self.E * np.exp(
            self.ax_history_.sel(Iteration=iteration - 1) + 
            self.bx_history_.sel(Iteration=iteration - 1) * 
            self.kt_history_.sel(Iteration=iteration - 1)
        )
        ax_new = (
            self.ax_history_.sel(Iteration=iteration - 1) + 
            (self.D - D_pred).sum(dim="Year") / D_pred.sum(dim="Year")
        )
        self.ax_history_.loc[{"Iteration": iteration}] = ax_new
        
    def _beta_update(self, iteration: int):
        D_pred = self.E * np.exp(
            self.ax_history_.sel(Iteration=iteration) + # Note the +1
            self.bx_history_.sel(Iteration=iteration - 1) * 
            self.kt_history_.sel(Iteration=iteration - 1)
        )
        bx_new = (
            self.bx_history_.sel(Iteration=iteration - 1) + 
            (self.kt_history_.sel(Iteration=iteration - 1) * (self.D - D_pred))
            .sum(dim="Year") / 
            (D_pred * self.kt_history_.sel(Iteration=iteration - 1)**2)
            .sum(dim="Year")
        )

        bx_new = bx_new / bx_new.sum()
        self.bx_history_.loc[{"Iteration": iteration}] = bx_new

    def _kappa_update(self, iteration: int):
        D_pred = self.E * np.exp(
            self.ax_history_.sel(Iteration=iteration) +
            self.bx_history_.sel(Iteration=iteration) * 
            self.kt_history_.sel(Iteration=iteration - 1)
        )
        kt_new = (
            self.kt_history_.sel(Iteration=iteration - 1) + 
            (self.bx_history_.sel(Iteration=iteration) * (self.D - D_pred))
            .sum(dim="Age") / 
            (D_pred * self.bx_history_.sel(Iteration=iteration)**2)
            .sum(dim="Age")
        )
        kt_new = kt_new - kt_new.mean()
        self.kt_history_.loc[{"Iteration": iteration}] = kt_new

    def predict(self, steps: int, simulations: int = 1) -> xr.DataArray:
        return steps