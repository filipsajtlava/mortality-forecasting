from abc import ABC, abstractmethod

import xarray as xr
import numpy as np

class GLMCapable(ABC):
    @abstractmethod
    def _compute_log_likelihood(self, *args, **kwargs) -> float:
        pass

    def get_pearson_residuals(self) -> xr.DataArray:
        D_pred = self.predict_in_sample() * self.E
        pearson_residuals = (self.D - D_pred) / np.sqrt(D_pred)
        return pearson_residuals

    def get_deviance_residuals(self):
        D_pred = self.predict_in_sample() * self.E

        deviance_residuals = np.sign(self.D - D_pred) * np.sqrt(2 * (
            self.D * np.log(self.D / D_pred) - (self.D - D_pred)
        ))
        return deviance_residuals