from pathlib import Path
from typing import Self

import pandas as pd
import numpy as np
import xarray as xr

import config
from core.commons import validate_value_column

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
            self.data.pivot(
                index=config.AGE_DIM, 
                columns=config.YEAR_DIM, 
                values=value_column
            )
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
            "start": int(self.data[config.YEAR_DIM].min()),
            "end": int(self.data[config.YEAR_DIM].max())
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
            query_str = f"{config.YEAR_DIM} <= {year}"
        else:
            query_str = (
                f"{config.YEAR_DIM} >= {year}" if overlap 
                else f"{config.YEAR_DIM} > {year}"
            )
        filtered_df = self.data.query(query_str).copy()
        return DemographicGrid(filtered_df, overlap)


class DemographicGridLoader:
    @classmethod
    def load_from_cached_file(
            cls,
            full_path: Path,
            starting_year: int,
            ending_year: int,
            maximum_age: int
        ) -> DemographicGrid:
        """Loads the data from .txt format with specified year intervals
        and a maximum possible age and then preprocesses it.
        
        Parameters
        ----------
        full_path
            Path to the cached .txt file.
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
        data[config.AGE_DIM] = (
            data[config.AGE_DIM]
            .astype(str).str
            .replace("+", "", regex=False)
            .astype(int)
        ) # We need to remove the "+" from 110+ to be able to use filters
        loading_query = (
            f"{config.YEAR_DIM} >= {starting_year} " +
            f"and {config.YEAR_DIM} <= {ending_year} and {config.AGE_DIM} <= {maximum_age}"
        )
        data = data.query(loading_query)
        preprocessed_data = cls.preprocessing(data)
        return DemographicGrid(preprocessed_data)

    @classmethod
    def manual_load_from_file(
            cls,
            input_dict: dict[str, str | Path | pd.DataFrame],
            preprocessing: bool
        ) -> DemographicGrid:
        dataframe_dict = cls._normalize_manual_input(input_dict)
        long_formats_to_concatenate = []
        for value_column, dataframe in dataframe_dict.items():
            series_long = (
                dataframe
                .stack(future_stack=True)
                .rename_axis(index=[config.AGE_DIM, config.YEAR_DIM])
                .rename(value_column)
            )
            long_formats_to_concatenate.append(series_long)

        cls._check_identical_age_year_structure(long_formats_to_concatenate)
        data = pd.concat(long_formats_to_concatenate, axis=1).reset_index()
        data[[config.AGE_DIM, config.YEAR_DIM]] = (
            data[[config.AGE_DIM, config.YEAR_DIM]].astype(int)
        )

        if preprocessing:
            data = cls.preprocessing(data)
        return DemographicGrid(data=data)

    @classmethod
    def _normalize_manual_input(
            cls,
            input_dict: dict[str, str | Path | pd.DataFrame]
        ) ->  dict[str, pd.DataFrame]:
        normalized = {}
        for value_column, user_input in input_dict.items():
            validate_value_column(value_column)
            if isinstance(user_input, (str, Path)):
                user_input = Path(user_input)
                if user_input.suffix == ".csv":
                    dataset = pd.read_csv(user_input, index_col=0)
                else:
                    raise ValueError(
                        "Please enter a valid path to a '.csv' file."
                    )
                normalized[value_column] = dataset
            else:
                normalized[value_column] = user_input
        return normalized

    @classmethod
    def _check_identical_age_year_structure(
        cls,
        datasets: list[pd.Series]
    ) -> None:
        reference_dataset = datasets[0]
        for dataset in datasets[1:]:
            if not reference_dataset.index.equals(dataset.index):
                raise ValueError(
                    f"The supplied data has different {config.AGE_DIM} and/or " 
                    f"{config.YEAR_DIM} dimensions."
                )    

    @classmethod
    def preprocessing(
            cls, 
            raw_df: pd.DataFrame
        ) -> pd.DataFrame:
        """Preprocessing of the loaded HMD .txt files using interpolation 
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
        target_columns = list(
            set(config.VALUE_COLUMNS).intersection(raw_df.columns)
        )
        df = raw_df.copy()

        df[target_columns] = df[target_columns].where(
            df[target_columns] != 0, 1e-9
        )
        df[target_columns] = np.log(df[target_columns])
        df = df.pivot(
            index=config.AGE_DIM, 
            columns=config.YEAR_DIM, 
            values=target_columns
        ).interpolate(method="linear", axis="index", limit_direction="both") 
        df = np.exp(df)

        df = df.stack(level=config.YEAR_DIM, future_stack=True).reset_index()
        return df