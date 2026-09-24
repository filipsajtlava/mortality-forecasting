from abc import ABC, abstractmethod

import numpy as np
import xarray as xr


class LikelihoodFamily(ABC):
    @abstractmethod
    def compute_log_likelihood(
            self, 
            D: xr.DataArray, 
            D_pred: xr.DataArray,
            **kwargs
        ) -> float:
        pass

    @abstractmethod
    def get_variance(
            self, 
            D_pred: xr.DataArray, 
            **kwargs
        ) -> xr.DataArray:
        pass

    @abstractmethod
    def get_deviance_residuals(
            self, 
            D: xr.DataArray, 
            D_pred: xr.DataArray,
            **kwargs
        ) -> xr.DataArray:
        pass

    @abstractmethod
    def numerical_optimization_factor(
            self,
            D: xr.DataArray, 
            D_pred: xr.DataArray,
            **kwargs
        ) -> tuple[xr.DataArray | float, xr.DataArray | float]:
        pass

    @abstractmethod
    def update_dispersion(
            self,
            D: xr.DataArray,
            D_pred: xr.DataArray,
            **kwargs
        ) -> float | None:
        pass


class Poisson(LikelihoodFamily):
    def compute_log_likelihood(
            self, 
            D: xr.DataArray, 
            D_pred: xr.DataArray,
            **kwargs
        ) -> float:
        return float((
            D * np.log(D_pred) - D_pred
        ).sum())

    def get_variance(self, D_pred: xr.DataArray, **kwargs) -> xr.DataArray:
        return D_pred

    def get_deviance_residuals(
            self, 
            D: xr.DataArray, 
            D_pred: xr.DataArray,
            **kwargs
        ) -> xr.DataArray:
        log_term = xr.where(D == 0, 0.0, np.log(D / D_pred))
        dev = 2 * (D * log_term - (D - D_pred))
        return np.sign(D - D_pred) * np.sqrt(np.maximum(dev, 0.0))

    def numerical_optimization_factor(
            self,
            D: xr.DataArray, 
            D_pred: xr.DataArray,
            **kwargs
        ) -> tuple[float, float]:
        return 1.0, 1.0

    def update_dispersion(
            self,
            D: xr.DataArray,
            D_pred: xr.DataArray,
            **kwargs
        ) -> None:
        return None


class NegativeBinomial(LikelihoodFamily):
    def compute_log_likelihood(
            self, 
            D: xr.DataArray, 
            D_pred: xr.DataArray,
            lambda_dispersion: float,
            **kwargs
        ) -> float:
        D_max = D.max()
        death_lookup_array = np.insert(
            np.cumsum(np.log(lambda_dispersion + np.arange(0, D_max))), 0, 0
        )
        return float((
            death_lookup_array[D.astype(int)] - (D + lambda_dispersion) * 
            np.log(1 + D_pred / lambda_dispersion) +
            D * np.log(D_pred / lambda_dispersion)
        ).sum())

    def get_variance(
            self, 
            D_pred: xr.DataArray, 
            lambda_dispersion: float,
            **kwargs
        ) -> xr.DataArray:
        return D_pred + (D_pred ** 2) / lambda_dispersion

    def get_deviance_residuals(
            self,
            D: xr.DataArray, 
            D_pred: xr.DataArray,
            lambda_dispersion: float,
            **kwargs
        ) -> xr.DataArray:
        log_term = xr.where(D == 0, 0.0, D * np.log(D / D_pred))
        dev = 2 * (
            log_term - (D + lambda_dispersion) * np.log(
                (D + lambda_dispersion) / 
                (D_pred + lambda_dispersion)
            )
        )
        return np.sign(D - D_pred) * np.sqrt(np.maximum(dev, 0.0))

    def numerical_optimization_factor(
            self,
            D: xr.DataArray, 
            D_pred: xr.DataArray,
            lambda_dispersion: float,
            **kwargs
        ) -> tuple[xr.DataArray, xr.DataArray]:
        num_factor = (lambda_dispersion + D) / (lambda_dispersion + D_pred)
        denom_factor = lambda_dispersion * (lambda_dispersion + D) / ((lambda_dispersion + D_pred) ** 2)
        return num_factor, denom_factor

    def update_dispersion(
            self,
            D: xr.DataArray,
            D_pred: xr.DataArray,
            lambda_dispersion: float,
            **kwargs
        ) -> float:
        max_death = D.max()
        D_sum_numerator = np.insert(
            np.cumsum(1 / (lambda_dispersion + np.arange(0, max_death))), 0, 0
        )
        D_sum_denominator = np.insert(
            np.cumsum(1 / (lambda_dispersion + np.arange(0, max_death)) ** 2), 0, 0
        )
        lambda_dispersion_new = float(
            lambda_dispersion + (
                D_sum_numerator[D.astype(int)] + 
                np.log(lambda_dispersion / (lambda_dispersion + D_pred)) + 
                (D_pred - D) / (D_pred + lambda_dispersion)
            ).sum() / (
                D_sum_denominator[D.astype(int)] - 
                1 / lambda_dispersion +
                (2 * D_pred + lambda_dispersion - D) / (lambda_dispersion + D_pred) ** 2 
            ).sum()
        )
        return lambda_dispersion_new