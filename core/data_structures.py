import pandas as pd
import numpy as np
import xarray as xr
from dataclasses import dataclass, field

@dataclass # Used solely for making navigation in the imported data and its history easier
class MortalityData:
    data: pd.DataFrame
    year_interval: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        if self.data is not None:
            self.year_interval = {
                "start": int(self.data["Year"].min()),
                "end": int(self.data["Year"].max())
            }
        else:
            self.year_interval = {
                "start": None,
                "end": None
            }

    def get_pivoted_data(self, value_column: str) -> xr.DataArray:
        """Pivots the data into a wide format widely used by different mortality methods.

        Parameters
        ----------
        value_column
            The specific column of values to use for the pivot.

        Returns
        -------
            Pivoted xr.DataArray in it's wide version.

        Raises
        ------
        ValueError
            If no data is present.
        """
        if self.data is None:
            raise ValueError("The passed mortality dataclass is empty.")
        
        return xr.DataArray(self.data.pivot(index="Age", columns="Year", values=value_column))

    def split_by_year(self, year: int, verbose: bool = True) -> tuple["MortalityData", "MortalityData"]:
        """Splits the data into two parts - training data (<= year) and testing data (> year).

        Parameters
        ----------
        year
            The year used for dividing the set into two parts.

        Returns
        -------
            A tuple of the two MortalityData instances, with the training one being first.
        """
        train_df = self.data.query(f"Year <= {year}")
        test_df = self.data.query(f"Year > {year}")

        if verbose:
            data_timeframe = self.year_interval["end"] - self.year_interval["start"] + 1
            train_years = year + 1 - self.year_interval["start"]
            test_years = self.year_interval["end"] - year
            train_portion = np.round(train_years * 100 / data_timeframe, 1)
            test_portion = np.round(test_years * 100 / data_timeframe, 1)

            print(f"Total years: {data_timeframe}")
            print(f"Training dataset: {train_years} ({train_portion} %)")
            print(f"Testing dataset: {test_years} ({test_portion} %) ")

        return MortalityData(train_df), MortalityData(test_df)