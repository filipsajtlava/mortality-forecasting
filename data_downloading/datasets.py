from typing import Self

import numpy as np
import xarray as xr
import pandas as pd

from data_downloading._hmd_data_fetcher import DataFetcherHMD
from data_downloading._loaders import DemographicGridLoader
from config import FILE_SELECTION_COUNTRY_DATA


class MortalityDataset:
    def __init__(self, country_code: str) -> None:
        self.country_code = country_code

    # TODO: add the ability to load data using a pairwise dictionary 
    def load_data(
            self,
            starting_year: int, 
            ending_year: int,
            metrics: list[str] | str | None = None, 
            maximum_age: int = 90
        ) -> Self:
        """Loads cached datasets or downloads data from the HMD database 
        filtering by the specified arguments.
        
        Parameters
        ----------
        starting_year
            First year included in the time frame.
        ending_year
            Last year included in the time frame.
        metrics, optional
            Select specific datagrids to load, with the year-interval
            applied on them. Can be either a list or a single string, by default None.
        maximum_age, optional
            Maximum age included in the dataset, by default 90.

        Returns
        -------
            The instance itself with loaded mortality data attributes.
        """

        self._initialize_and_validate()

        user_selection = self._normalize_metrics(metrics)
        successfully_loaded = self._data_fetcher.fetch_country_data()

        for metric in user_selection:
            if metric in successfully_loaded:
                new_demo_grid = DemographicGridLoader.load_from_file(
                    successfully_loaded[metric],
                    starting_year,
                    ending_year,
                    maximum_age
                )
                setattr(
                    self,
                    metric,
                    new_demo_grid
                )
        return self

    def train_test_split(
            self, 
            year: int,
            metrics: list[str] | str | None = None, 
            overlap: bool = False
        ) -> tuple[Self, Self]:
        """Splits the data into two parts - 
        training data (<= year) and testing data (> year).

        Parameters
        ----------
        year
            The year used for dividing the set into two parts.
        metrics, optional
            Select specific datagrids to split into training and testing portions.
            Can be either a list or a single string, by default None.
        overlap, optional
            If set to True, the last year of the training set will be the 
            same as the first year of the testing set, by default False.

        Returns
        -------
            A tuple of the two MortalityDataset instances, 
            with the training one being first.
        """
        if not hasattr(self, "_data_fetcher"):
            raise ValueError("Cannot split dataset, no data has been loaded yet.")

        train_ds = MortalityDataset(self.country_code)
        test_ds = MortalityDataset(self.country_code)
        train_ds._initialize_and_validate()
        test_ds._initialize_and_validate()

        user_selection = self._normalize_metrics(metrics)
        for metric in user_selection:
            demo_grid = getattr(self, metric, None)
            if demo_grid is not None:
                grid_year_span = np.arange(
                    demo_grid.year_interval["start"],
                    demo_grid.year_interval["end"] + 1
                )
                if year not in grid_year_span:
                    raise ValueError("The selected splitting year is invalid.")
                
                setattr(
                    train_ds,
                    metric,
                    demo_grid._filter_by_year(year, is_train=True, overlap=overlap)
                )
                setattr(
                    test_ds,
                    metric,
                    demo_grid._filter_by_year(year, is_train=False, overlap=overlap)
                )
        return (train_ds, test_ds)

    def info(self) -> None:
        """Prints information about the currently loaded metrics in the 
        dataset.
        """
        for metric, file_name in FILE_SELECTION_COUNTRY_DATA.items():
            print(f"{file_name}:")
            demo_grid = getattr(self, metric, None)
            if demo_grid:
                print(
                    f"{" "*5}The grid '{metric}' is loaded in the timespan " \
                    f"{demo_grid.year_interval["start"]}-{demo_grid.year_interval["end"]}."
                )
            else:
                print(f"{" "*5}Currently not loaded.")

    def _normalize_metrics(self, metrics: list[str] | str | None) -> list[str]:
        """Normalizes the entered metrics into a single list containing the user
        selection in accordance to the available ones.

        Parameters
        ----------
        metrics, optional
            Select specific datagrids to load, with the year-interval
            applied on them. Can be either a list or a single string, by default None.
        
        Returns
        -------
        user_selection
            Normalized list of metrics.
        """
        valid_options = list(FILE_SELECTION_COUNTRY_DATA.keys())
        if metrics is None:
            user_selection = valid_options
        elif isinstance(metrics, str) and metrics in valid_options:
            user_selection = [metrics]
        elif (
            isinstance(metrics, (list, tuple, set)) and not
            (set(metrics) - set(valid_options)) # checks if user input is available
        ):
            user_selection = list(set(metrics))
        else:
            raise ValueError("The entered metrics are invalid")
        return user_selection

    def _initialize_and_validate(self) -> None:
        """Initializes the HMD data fetcher in the case that it was not initialized
        during an earlier load_data run. Also validates the country code and sets
        individual metrics to None.
        """
        if hasattr(self, "_data_fetcher"):
            return

        for metric in FILE_SELECTION_COUNTRY_DATA.keys():
            setattr(self, metric, None)

        fetcher = DataFetcherHMD(self.country_code)

        if not fetcher.is_country_code_valid():
            raise ValueError(f"Selected country code '{self.country_code}' is invalid")
        self._data_fetcher = fetcher


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
