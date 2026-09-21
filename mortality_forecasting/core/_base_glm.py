from abc import ABC, abstractmethod

import xarray as xr
import numpy as np

class GLMCapable(ABC):
    @abstractmethod
    def _compute_log_likelihood(self, *args, **kwargs) -> float:
        pass

    @abstractmethod
    def _variance(self, deaths: xr.DataArray) -> xr.DataArray:
        pass

    @property
    def pearson_residuals(self) -> xr.DataArray:
        D_pred = self.predict_in_sample() * self.E
        pearson_residuals = (self.D - D_pred) / np.sqrt(self._variance(D_pred))
        return pearson_residuals

    @property
    @abstractmethod
    def deviance_residuals(self) -> xr.DataArray:
        pass