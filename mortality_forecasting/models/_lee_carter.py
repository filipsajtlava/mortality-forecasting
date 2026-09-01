from typing import Self

import numpy as np
import xarray as xr

from mortality_forecasting.data_processing._dataset import MortalityDataset
from mortality_forecasting.core._base_model import Model
from mortality_forecasting.core._commons import (
    validate_value_column, 
    ParameterContainer
)
from mortality_forecasting import config


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
        self._validate_dataset(
            mortality_data=mortality_data,
            value_column=value_column
        )

        self.mortality_data = mortality_data
        self.value_column = value_column

        M = (
            self.mortality_data.D[self.value_column] /
            self.mortality_data.E[self.value_column]
        )

        if self.lee_miller_fix:
            ax_ = np.log(M).sel({
                config.YEAR_DIM: self.mortality_data.D.year_interval["end"]
            })
        else:
            ax_ = np.log(M).mean(axis=1)
        Z_centered = np.log(M) - ax_

        U, s, V = np.linalg.svd(Z_centered.values, full_matrices=False)
        
        scaling_factor = U[:, 0].sum()
        bx_ = xr.DataArray(
            U[:, 0] / scaling_factor, 
            coords=[(config.AGE_DIM, np.log(M)[config.AGE_DIM].values)]
        )
        kt_ = xr.DataArray(
            s[0] * V[0, :] * scaling_factor, 
            coords=[(config.YEAR_DIM, np.log(M)[config.YEAR_DIM].values)]
        )

        self.parameters_ = ParameterContainer(
            static=xr.Dataset(
                data_vars={
                    "ax": ax_,
                    "bx": bx_
                }
            ),
            period=xr.Dataset(
                data_vars={
                    "kt": kt_
                },                
                attrs={
                    "overlap": self.mortality_data.D.overlap,
                    "last_year": self.mortality_data.D.year_interval["end"]
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