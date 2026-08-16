from pathlib import Path
from typing import Self

import pandas as pd
import numpy as np
import xarray as xr

from config import VALUE_COLUMNS


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
        """Computes the maximum and minimum year in the data.
        
        Returns
        -------
            A dictionary containing the 'start' and 'end' years.
        """
        year_interval = {
            "start": int(self.data["Year"].min()),
            "end": int(self.data["Year"].max())
        }
        return year_interval

    def _filter_by_year(self, year: int, is_train: bool, overlap: bool) -> Self:
        """Filters the data by the selected year depending 
        on the chosen parameters.
        
        Parameters
        ----------
        year
            Year under (or over) which the method filters the dataset.
        is_train
            Boolean value dictating if the dataset is supposed to be training
            or testing (training == years before the selected 'year').
        overlap
            Boolean value determining whether the boundary 'year' itself is included
            in both the training and testing splits ('<='/'>' vs '<='/'=>').

        Returns
        -------
            Filtered DemographicGrid instance.
        """
        if is_train:
            query_str = f"Year <= {year}"
        else:
            query_str = f"Year >= {year}" if overlap else f"Year > {year}"
        filtered_df = self.data.query(query_str).copy()
        return DemographicGrid(filtered_df, overlap)


class DemographicGridLoader:
    @classmethod
    def load_from_file(
            cls,
            full_path: Path,
            starting_year: int,
            ending_year: int,
            maximum_age: int
        ) -> DemographicGrid:
        """Loads the data from .csv format with specified year intervals
        and a maximum possible age and then preprocesses it.
        
        Parameters
        ----------
        full_path
            Path to the cached .csv file.
        starting_year
            First year included in the time frame.
        ending_year
            Last year included in the time frame.
        maximum_age
            Maximum age included in the dataset.

        Returns
        -------
            A loaded DemographicGrid instance. 
        """
        data = pd.read_csv(full_path, sep=r"\s+", header=1, na_values=".")
        data["Age"] = (
            data["Age"]
            .astype(str).str
            .replace("+", "", regex=False)
            .astype(int)
        ) # We need to remove the "+" from 110+ to be able to use filters
        loading_query = (
            f"Year >= {starting_year} " +
            f"and Year <= {ending_year} and Age <= {maximum_age}"
        )
        data = data.query(loading_query)
        preprocessed_data = cls.preprocessing(data)
        return DemographicGrid(preprocessed_data)

    @classmethod
    def preprocessing(
            cls, 
            raw_df: pd.DataFrame
        ) -> pd.DataFrame:
        """Preprocessing of the loaded HMD .csv files using interpolation 
        and filling empty values with very small numbers which allow
        for using logarithms.

        Parameters
        ----------
        raw_df
            Loaded dataset ready to be preprocessed.

        Returns
        -------
            Preprocessed dataset.
        """
        target_columns = VALUE_COLUMNS.copy()
        df = raw_df.copy()

        df[target_columns] = df[target_columns].where(
            df[target_columns] != 0, 1e-9
        )
        df[target_columns] = np.log(df[target_columns])
        df = df.pivot(
            index="Year", 
            columns="Age", 
            values=target_columns
        ).interpolate(method="linear", axis=0, limit_direction="both") 
        df = np.exp(df)

        df = df.stack(level="Age", future_stack=True).reset_index()
        return df