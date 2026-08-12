from typing import Self
import pandas as pd
import numpy as np
import xarray as xr

class DemographicGrid:
    def __init__(
            self, 
            data: pd.DataFrame, 
            overlap: bool = False
        ) -> None:

        self.data = data
        self.overlap = overlap
        self.year_interval = self._compute_year_interval()


    def __getitem__(self, value_column: str) -> xr.DataArray:
        return self._get_pivoted_data(value_column)


    @property
    def Female(self) -> xr.DataArray:
        return self["Female"]


    @property
    def Male(self) -> xr.DataArray:
        return self["Male"]


    @property
    def Total(self) -> xr.DataArray:
        return self["Total"]


    def _compute_year_interval(self) -> dict[str, int]:
        year_interval = {
            "start": int(self.data["Year"].min()),
            "end": int(self.data["Year"].max())
        }
        return year_interval


    def _get_pivoted_data(self, value_column: str) -> xr.DataArray:
        """Pivots the data into a wide matrix format 
        widely used by different mortality methods.

        Parameters
        ----------
        value_column
            The specific column of values to use for the pivot.

        Returns
        -------
            Pivoted xr.DataArray in it's wide version.
        """
        return xr.DataArray(
            self.data.pivot(index="Age", columns="Year", values=value_column)
        )


    # TODO: the user should never have access to DemograpicGrid, either make
    # this a private method, or return two MortalityDataset instances
    def split_by_year(
            self, 
            year: int, 
            overlap: bool = False, 
            verbose: bool = True
        ) -> tuple[Self, Self]:
        """Splits the data into two parts - 
        training data (<= year) and testing data (> year).

        Parameters
        ----------
        year
            The year used for dividing the set into two parts.
        overlap, optional
            If set to True, the last year of the training set will be the 
            same as the first year of the testing set.
        verbose, optional
            Print details about the split.

        Returns
        -------
            A tuple of the two DemographicGrid instances, 
            with the training one being first.
        """
        train_df = self.data.query(f"Year <= {year}")

        if overlap:
            test_df = self.data.query(f"Year >= {year}")
        else:
            test_df = self.data.query(f"Year > {year}")

        if verbose:
            data_timeframe = self.year_interval["end"] - self.year_interval["start"] + 1
            train_years = year + 1 - self.year_interval["start"]
            if overlap:
                test_years = self.year_interval["end"] - year + 1
            else:
                test_years = self.year_interval["end"] - year
            train_portion = np.round(train_years * 100 / data_timeframe, 1)
            test_portion = np.round(test_years * 100 / data_timeframe, 1)

            print(f"Total years: {data_timeframe}")
            print(f"Training dataset: {train_years} ({train_portion} %)")
            print(f"Testing dataset: {test_years} ({test_portion} %) ")
            
        return (
            DemographicGrid(data=train_df, overlap=overlap), 
            DemographicGrid(data=test_df, overlap=overlap)
        )