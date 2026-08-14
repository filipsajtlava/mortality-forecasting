from typing import Self

import numpy as np
import xarray as xr

from data_downloading.datasets import MortalityDataset
from core.base_model import Model

class LeeCarterModel(Model):
    # TODO: add simulations to the __init__
    def __init__(
            self,
            lee_miller_fix: bool = False,
            seed: int | np.random.Generator | None = None
        ) -> None:
        super().__init__(lee_miller_fix=lee_miller_fix, seed=seed)

    def fit(self, mortality_data: MortalityDataset, value_column: str) -> Self:
        """Fit the Lee-Carter model using SVD.
        """
        self.mortality_data = mortality_data
        self.value_column = value_column
        self._validate_dataset(
            mortality_data=self.mortality_data,
            required_grids=["M"]
        )
        self.overlap_step = 0 if self.mortality_data.M.overlap else 1

        log_M = np.log(self.mortality_data.M[self.value_column])

        if self.lee_miller_fix:
            ax_estimate_da = log_M.sel(Year=self.mortality_data.M.year_interval["end"])
        else:
            ax_estimate_da = log_M.mean(axis=1)
        self.ax_ = ax_estimate_da.values
        Z_centered = log_M - ax_estimate_da

        U, s, V = np.linalg.svd(Z_centered.values, full_matrices=False)
        
        scaling_factor = U[:, 0].sum()
        self.bx_ = U[:, 0] / scaling_factor
        self.kt_ = s[0] * V[0, :] * scaling_factor

        self.explained_variance_ = s[0]**2 / np.sum(s**2)
        self.drift_ = (self.kt_[-1] - self.kt_[0]) / (len(self.kt_) - 1)
        self.std_of_errors_ = np.std(np.diff(self.kt_) - self.drift_)
        return self

    def forecast_kt(self, steps: int, simulations: int) -> xr.DataArray:
        """Simulate and forecast the stochastic walk of the kt parameter.
        """
        rng = self._normalize_seed()
        innovations = rng.normal(
            self.drift_, self.std_of_errors_, size=(steps, simulations)
        )
        innovations = np.insert(innovations, 0, 0, axis=0)
        kt_forecast = self.kt_[-1] + np.cumsum(innovations, axis=0)

        last_year = self.mortality_data.M.year_interval["end"]
        pred_years = np.arange(self.overlap_step, steps + 1) + last_year

        return xr.DataArray(
            kt_forecast[self.overlap_step:],
            coords=[pred_years, np.arange(1, simulations + 1)],
            dims=["Year", "Simulation"],
            name="kt_forecast"
        )
    
    def forecast_kt_analytical(self, steps: int) -> xr.DataArray:
        """Forecast the values of the kt parameter analytically.
        """
        drifts = np.repeat(self.drift_, steps)
        drifts = np.insert(drifts, 0, 0)
        kt_forecast = self.kt_[-1] + np.cumsum(drifts, axis=0)

        last_year = self.mortality_data.M.year_interval["end"]
        pred_years = np.arange(self.overlap_step, steps + 1) + last_year

        return xr.DataArray(
            kt_forecast[self.overlap_step:],
            coords=[pred_years],
            dims=["Year"],
            name="kt_forecast_analytical"
        )

    def predict(self, steps: int, simulations: int = 1, stochastic: bool = True) -> xr.DataArray:
        """Predict the future mortality values.
        """
        if not stochastic and simulations > 1:
            simulations_warning = "WARNING: setting simulations to a higher number " \
            "than one while using analytical forecasts is without effect."
            print(simulations_warning)

        if stochastic:
            kt_preds = self.forecast_kt(steps, simulations)
            output_data_name = "M_forecast"
        else:
            kt_preds = self.forecast_kt_analytical(steps)
            output_data_name = "M_forecast_analytical"

        ages = self.mortality_data.M[self.value_column].coords["Age"].values
        log_M_preds = xr.DataArray(self.ax_, dims="Age") + \
            xr.DataArray(self.bx_, dims="Age") * kt_preds
        log_M_preds = log_M_preds.assign_coords(Age=ages)
        return np.exp(log_M_preds).rename(output_data_name)

    # TODO: this should be completely removed and the user should compute it themselves
    def predict_historical(self):
        log_M_preds_historical = self.ax_[:, np.newaxis] + self.bx_[:, np.newaxis] * self.kt_
        
        return xr.DataArray(
            np.exp(log_M_preds_historical),
            coords=[
                self.mortality_data.M[self.value_column].to_pandas().index.to_numpy(), 
                self.mortality_data.M[self.value_column].to_pandas().columns.to_numpy()
            ],
            dims=["Age", "Year"],
            name="mx_historical"
        )