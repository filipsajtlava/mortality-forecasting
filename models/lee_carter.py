from typing import Self

import numpy as np
import xarray as xr

from data_downloading.datasets import MortalityDataset
from core.base_model import Model
from core.commons import validate_value_column, ParameterContainer
import config

class LeeCarterModel(Model):
    def __init__(
            self,
            lee_miller_fix: bool = False
        ) -> None:
        super().__init__(lee_miller_fix=lee_miller_fix)

    def fit(self, mortality_data: MortalityDataset, value_column: str) -> Self:
        """Fit the Lee-Carter model using SVD.
        """
        validate_value_column(value_column)
        required_grids = ("M", )
        self._validate_dataset(
            mortality_data=mortality_data,
            required_grids=required_grids
        )

        self.mortality_data = mortality_data
        self.value_column = value_column

        log_M = np.log(self.mortality_data.M[self.value_column])

        if self.lee_miller_fix:
            self.ax_ = log_M.sel({
                config.YEAR_DIM: self.mortality_data.M.year_interval["end"]
            })
        else:
            self.ax_ = log_M.mean(axis=1)
        Z_centered = log_M - self.ax_

        U, s, V = np.linalg.svd(Z_centered.values, full_matrices=False)
        
        scaling_factor = U[:, 0].sum()
        self.bx_ = xr.DataArray(
            U[:, 0] / scaling_factor, 
            coords=[(config.AGE_DIM, log_M[config.AGE_DIM].values)]
        )
        self.kt_ = xr.DataArray(
            s[0] * V[0, :] * scaling_factor, 
            coords=[(config.YEAR_DIM, log_M[config.YEAR_DIM].values)]
        )

        self.parameters_ = ParameterContainer(
            static=xr.Dataset(
                data_vars={
                    "ax": self.ax_,
                    "bx": self.bx_
                }
            ),
            period=xr.Dataset(
                data_vars={
                    "kt": self.kt_
                },                
                attrs={
                    "overlap": self.mortality_data.M.overlap,
                    "last_year": self.mortality_data.M.year_interval["end"]
                }
            )
        )

        self.explained_variance_ = s[0]**2 / np.sum(s**2)
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