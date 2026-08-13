from pathlib import Path
from typing import Self
import pandas as pd
import numpy as np
from core._data_structures import DemographicGrid
from data_downloading._hmd_data_fetcher import DataFetcherHMD
from config import (
    FILE_SELECTION_COUNTRY_DATA,
    DATA_DIRECTORY_NAME
)


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
        self
            The instance itself with loaded mortality data attributes.
        """

        self._initialize_and_validate()

        user_selection = self._normalize_metrics(metrics)
        successfully_loaded = self._data_fetcher.fetch_country_data()

        for metric in user_selection:
            if metric in successfully_loaded:
                setattr(
                    self,
                    metric, 
                    self._preprocessing(
                        successfully_loaded[metric], 
                        starting_year, 
                        ending_year, 
                        maximum_age
                    )
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
                    demo_grid.filter_by_year(year, is_train=True, overlap=overlap)
                )
                setattr(
                    test_ds,
                    metric,
                    demo_grid.filter_by_year(year, is_train=False, overlap=overlap)
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

        hmd_data_path = Path.cwd() / DATA_DIRECTORY_NAME 
        fetcher = DataFetcherHMD(hmd_data_path, self.country_code)

        if not fetcher.is_country_code_valid():
            raise ValueError(f"Selected country code '{self.country_code}' is invalid")
        self._data_fetcher = fetcher


    def _preprocessing(
            self, 
            full_path: str, 
            starting_year: int,
            ending_year: int, 
            maximum_age: int
        ) -> DemographicGrid:
        """Minimal preprocessing of the downloaded HMD files.

        Returns
        -------
            MortalityData instance holding all of the data and the additional information together in one place
        """
        data = pd.read_csv(full_path, sep=r"\s+", header=1, na_values=".")
        data["Age"] = data["Age"].astype(str).str.replace("+", "", regex=False).astype(int) # We need to remove the "+" from 110+ to be able to use filters
        data = data.query(f"Year >= {starting_year} and Year <= {ending_year} and Age <= {maximum_age}")
        # TODO: This has to be redesigned with log values, to account for Gompertzs law
        target_columns = ["Female", "Male", "Total"]
        pivoted_values = data.pivot(
            index="Year", 
            columns="Age", 
            values=target_columns
        ).interpolate(method="linear", axis=0, limit_direction="both") 

        interpolated_data = pivoted_values.stack(level="Age").reset_index()
        interpolated_data[target_columns] = interpolated_data[target_columns].where(
            interpolated_data[target_columns] != 0, 1e-9
        )

        return DemographicGrid(interpolated_data)