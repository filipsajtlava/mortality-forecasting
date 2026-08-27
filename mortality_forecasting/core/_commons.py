from dataclasses import dataclass, fields
from typing import Iterator
from itertools import chain
import xarray as xr

from mortality_forecasting import config


def validate_value_column(value_column: str) -> None:
    if value_column not in config.VALUE_COLUMNS:
        raise ValueError(
            f"The selected value column '{value_column}' is unavailable, " \
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

    def info(self) -> None:
        for field in fields(self):
            ds = getattr(self, field.name)
            print(f"{field.name} parameters:")
            if ds is None:
                print(f"{config.INFO_INDENT}empty")
            else:
                for parameter in ds:
                    print(f"{config.INFO_INDENT}['{parameter}'] with {ds.coords}")
                
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
    mortality_rates_: xr.DataArray
    parameters_: ParameterContainer