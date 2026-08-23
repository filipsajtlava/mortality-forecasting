from dataclasses import dataclass
from typing import Iterator
from itertools import chain

import xarray as xr

import config


# TODO: I dont really know why this is here and not in the base model?
# maybe I was expecting it to be used somewhere else aswell
def validate_value_column(value_column: str) -> None:
    if value_column not in config.VALUE_COLUMNS:
        raise ValueError(
            f"The selected value column is unavailable, " \
            f"try one of the following: {config.VALUE_COLUMNS}"
        )


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

    def __iter__(self) -> Iterator[str]:
        return chain.from_iterable([
            ds for ds in (self.static, self.period, self.cohort) 
            if ds is not None
        ])

    def __len__(self) -> int:
        return sum(
            len(ds) for ds in (self.static, self.period, self.cohort)
            if ds is not None 
        )

    def __getitem__(self, parameter: str):
        for ds in (self.static, self.period, self.cohort):
            if ds is not None and parameter in ds:
                return ds[parameter]

        raise KeyError(f"Parameter '{parameter}' not found in the parameter container.")

@dataclass
class ForecastContainer:
    """A container class for forecasted parameters along with the mortality,
    rates, allowing the user and plotting devices to access the same 
    exact structure everywhere.

    Parameters
    ----------
    static
        Parameters which are not forecasted.
    period
        Forecasted period-like parameters.
    cohort, optional
        Forecastes cohort-like parameters, by default None.
    """
    mortality_rates: xr.DataArray
    parameters: ParameterContainer