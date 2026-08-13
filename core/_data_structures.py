from typing import Self
import pandas as pd
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


    def filter_by_year(self, year: int, is_train: bool, overlap: bool):
        if is_train:
            query_str = f"Year <= {year}"
        else:
            query_str = f"Year >= {year}" if overlap else f"Year > {year}"
        filtered_df = self.data.query(query_str).copy()
        return DemographicGrid(filtered_df, overlap=overlap)


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