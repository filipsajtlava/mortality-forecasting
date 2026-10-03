from dataclasses import dataclass, fields
from typing import Iterator, Literal, Any
from itertools import chain

import xarray as xr

from mortality_forecasting import config


# This was moved here from model plotter in case anything else uses it
def assert_glm(target: object) -> None:
    method = getattr(
        getattr(target, "model", target), "method", None
    )

    if method not in ("poisson", "negative_binomial"):
        raise TypeError(
            f"This operation requires a GLM-fitted model, but received '{method}'."
        )

def validate_value_column(value_column: str) -> None:
    if value_column not in config.VALUE_COLUMNS:
        raise ValueError(
            f"The selected value column '{value_column}' is unavailable, " \
            f"try one of the following: {config.VALUE_COLUMNS}"
        )

def bounds_from_simulations(
        da: xr.DataArray,
        alpha: float = 0.05,
        point_estimate: Literal["mean", "median"] = "median"
    ) -> xr.DataArray:

    lower_da = da.quantile(
        alpha / 2.0,
        dim=config.SIMULATION_DIM
    ).drop_vars("quantile", errors="ignore")
    point_da = getattr(da, point_estimate)(
        dim=config.SIMULATION_DIM
    )
    upper_da = da.quantile(
        1 - alpha / 2.0,
        dim=config.SIMULATION_DIM
    ).drop_vars("quantile", errors="ignore")
    bounds_da = (
        xr.concat([lower_da, point_da, upper_da], dim=config.BOUND_DIM)
        .assign_coords({config.BOUND_DIM: ["lower", "point", "upper"]})
        .transpose(config.AGE_DIM, config.YEAR_DIM, config.BOUND_DIM, ..., missing_dims="ignore")
    )
    return bounds_da

@dataclass
class ParameterContainer:
    """A container class for sorting parameters into 3 different groups,
    allowing forecasters to access the same exact structure everywhere.

    Parameters
    ----------
    static
        Parameters which are not forecasted.
    period
        Forecasted period-like parameters.
    cohort, optional
        Forecastes cohort-like parameters, by default None.
    """
    static: xr.Dataset
    period: xr.Dataset
    cohort: xr.Dataset | None = None

    @property
    def _datasets(self) -> tuple[xr.Dataset, ...]:
        return tuple(
            ds for field in fields(self)
            if (ds := getattr(self, field.name)) is not None
        )

    def __iter__(self) -> Iterator[str]:
        return chain.from_iterable(self._datasets)

    def __len__(self) -> int:
        return sum(len(ds) for ds in self._datasets)

    def __getitem__(self, parameter_selection: str) -> xr.DataArray:
        for ds in self._datasets:
            if parameter_selection in ds:
                return ds[parameter_selection]

        raise KeyError(f"Parameter '{parameter_selection}' not found.")

    def get(
            self, 
            parameter_selection: str, 
            default: Any = None
        ) -> xr.DataArray | Any:
        try:
            return self[parameter_selection]
        except KeyError:
            return default

    # TODO: rename this..
    def _indexed_items(self) -> Iterator[tuple[str, xr.DataArray]]:
        """Gives only parameters that are not singular scalars"""
        for ds in self._datasets:
            for name in ds:
                da = ds[name]
                if da.ndim > 0:
                    yield name, da

    @property
    def data(self) -> xr.Dataset:
        return xr.merge(self._datasets)
                
@dataclass
class ForecastContainer:
    """A container class for forecasted parameters along with the mortality,
    rates, allowing the user and plotting devices to access the same 
    exact structure everywhere.
    """
    mortality_rates_: xr.DataArray
    parameters_: ParameterContainer
    attrs: dict | None = None