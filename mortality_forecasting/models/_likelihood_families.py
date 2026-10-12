from abc import ABC, abstractmethod

import numpy as np
import xarray as xr
from scipy.special import gammaln, digamma, polygamma

from mortality_forecasting import config


class LikelihoodFamily(ABC):
    @abstractmethod
    def compute_log_likelihood(
            self, 
            D: xr.DataArray,
            E: xr.DataArray,
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
    def dispersion_initialization(self, **kwargs) -> xr.DataArray | None:
        pass

    @abstractmethod
    def update_dispersion(
            self,
            D: xr.DataArray,
            D_pred: xr.DataArray,
            **kwargs
        ) -> xr.DataArray | None:
        pass

    @abstractmethod
    def sample_from_distribution(
            self, 
            D_pred: xr.DataArray, 
            seed: np.random.Generator,
            **kwargs
        ) -> xr.DataArray:
        pass


class Poisson(LikelihoodFamily):
    def compute_log_likelihood(
            self, 
            D: xr.DataArray, 
            E: xr.DataArray,
            D_pred: xr.DataArray,
            **kwargs
        ) -> float:
        return (
            D * np.log(D_pred / E) - D_pred
        ).sum(dim=[config.AGE_DIM, config.YEAR_DIM])

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

    def dispersion_initialization(
            self, 
            bound_dims: dict[str, np.ndarray], 
            **kwargs
        ) -> xr.DataArray:
        shape = tuple(len(value) for value in bound_dims.values())
        return xr.DataArray(
            np.full(shape, None),
            coords=bound_dims
        )

    def update_dispersion(
            self,
            D: xr.DataArray,
            D_pred: xr.DataArray,
            **kwargs
        ) -> None:
        return None

    def sample_from_distribution(
            self, 
            D_pred: xr.DataArray, 
            seed: np.random.Generator,
            **kwargs
        ) -> xr.DataArray:
        return xr.apply_ufunc(
            seed.poisson,
            D_pred,
            keep_attrs=True
        )


class NegativeBinomial(LikelihoodFamily):
    def compute_log_likelihood(
            self, 
            D: xr.DataArray,
            E: xr.DataArray,
            D_pred: xr.DataArray,
            lambda_dispersion: xr.DataArray,
            **kwargs
        ) -> xr.DataArray:
        # this represents the \sum_{i=1}^{D_{x,t}} \ln(\lambda + D_{x,t} - i)
        log_gamma_term = gammaln(D + lambda_dispersion) - gammaln(lambda_dispersion)
        return (
            log_gamma_term - (D + lambda_dispersion) * 
            np.log(1 + D_pred / lambda_dispersion) +
            D * np.log(D_pred / lambda_dispersion)
        ).sum(dim=[config.AGE_DIM, config.YEAR_DIM])

    def get_variance(
            self, 
            D_pred: xr.DataArray, 
            lambda_dispersion: xr.DataArray,
            **kwargs
        ) -> xr.DataArray:
        return D_pred + (D_pred ** 2) / lambda_dispersion

    def get_deviance_residuals(
            self,
            D: xr.DataArray, 
            D_pred: xr.DataArray,
            lambda_dispersion: xr.DataArray,
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
            lambda_dispersion: xr.DataArray,
            **kwargs
        ) -> tuple[xr.DataArray, xr.DataArray]:
        num_factor = (lambda_dispersion + D) / (lambda_dispersion + D_pred)
        denom_factor = (
            lambda_dispersion * (lambda_dispersion + D) / 
            ((lambda_dispersion + D_pred) ** 2)
        )
        return num_factor, denom_factor

    def dispersion_initialization(
            self, 
            bound_dims: dict[str, np.ndarray],
            init: xr.DataArray | None, 
            **kwargs
        ) -> xr.DataArray:
        if isinstance(init, xr.DataArray):
            return init
        else:
            shape = tuple(len(value) for value in bound_dims.values())
            return xr.DataArray(
                np.full(shape, config.LAMBDA_INITIALIZATION),
                coords=bound_dims
            )

    def update_dispersion(
            self,
            D: xr.DataArray,
            D_pred: xr.DataArray,
            lambda_dispersion: xr.DataArray,
            **kwargs
        ) -> xr.DataArray:
        lambda_broadcasted = lambda_dispersion.broadcast_like(D)
        num_gamma = digamma(D + lambda_broadcasted) - digamma(lambda_broadcasted)
        denom_gamma = polygamma(1, lambda_broadcasted) - polygamma(1, D + lambda_broadcasted)

        D_sum_numerator = (
            num_gamma
            + np.log(lambda_broadcasted / (lambda_broadcasted + D_pred))
            + (D_pred - D) / (D_pred + lambda_broadcasted)
        ).sum(dim=[config.AGE_DIM, config.YEAR_DIM])
        D_sum_denominator = (
            denom_gamma
            - 1 / lambda_broadcasted
            + (2 * D_pred + lambda_broadcasted - D)
            / (lambda_broadcasted + D_pred) ** 2
        ).sum(dim=[config.AGE_DIM, config.YEAR_DIM])

        lambda_dispersion_new = lambda_dispersion + (
            D_sum_numerator / D_sum_denominator
        )
        return lambda_dispersion_new.clip(min=1e-2)

    def sample_from_distribution(
            self, 
            D_pred: xr.DataArray,
            seed: np.random.Generator,
            lambda_dispersion: xr.DataArray,
            **kwargs
        ) -> xr.DataArray:
        return xr.apply_ufunc(
            seed.negative_binomial,
            lambda_dispersion,
            lambda_dispersion / (lambda_dispersion + D_pred),
            keep_attrs=True,
        )