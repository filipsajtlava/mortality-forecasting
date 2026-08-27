from typing import Self
from pathlib import Path

import numpy as np
import pandas as pd

from ._hmd_data_fetcher import DataFetcherHMD
from ._grid import DemographicGridLoader
from mortality_forecasting import config


class MortalityDataset:
    def __init__(self, country_code: str) -> None:
        self.country_code = country_code

    def load_data(
            self,
            starting_year: int, 
            ending_year: int,
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
        maximum_age, optional
            Maximum age included in the dataset, by default 90.

        Returns
        -------
            The instance itself with loaded mortality data attributes.
        """

        self._initialize_and_validate()
        successfully_loaded = self._data_fetcher.fetch_country_data()

        for metric in config.FILE_SELECTION_COUNTRY_DATA.keys():
            if metric in successfully_loaded:
                new_demo_grid = DemographicGridLoader.load_from_cached_file(
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
            overlap: bool = False
        ) -> tuple[Self, Self]:
        """Splits the data into two parts - 
        training data (<= year) and testing data (> year).

        Parameters
        ----------
        year
            The year used for dividing the set into two parts.
        overlap, optional
            If set to True, the last year of the training set will be the 
            same as the first year of the testing set, by default False.

        Returns
        -------
            A tuple of the two MortalityDataset instances, 
            with the training one being first.
        """
        train_ds = MortalityDataset(self.country_code)
        test_ds = MortalityDataset(self.country_code)

        for metric in config.FILE_SELECTION_COUNTRY_DATA.keys():
            setattr(train_ds, metric, None)
            setattr(test_ds, metric, None)

        for metric in config.FILE_SELECTION_COUNTRY_DATA.keys():
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
        for metric, file_name in config.FILE_SELECTION_COUNTRY_DATA.items():
            print(f"{file_name}:")
            demo_grid = getattr(self, metric, None)
            if demo_grid:
                print(
                    f"{config.INFO_INDENT}The grid '{metric}' is loaded in the timespan " \
                    f"{demo_grid.year_interval['start']}-{demo_grid.year_interval['end']}."
                )
            else:
                print(f"{config.INFO_INDENT}Currently not loaded.")

    # TODO: this needs a comprehensive docstring explaining how the .csv files should
    # look, in the case that the Path appraoch is chosen - first column index Age
    # and names of columns are individual years
    # Maybe adding a quick example would be a good choice, like an actual table
    # TODO: this entire datasets.py file is completely modular, with the inputs
    # depending on the config file and the FILE SELECTION COUNTRY DATA
    # this was done in the case of anything being added, as it allows for streamlining
    # new grids. If nothing else is going to be used, this approach can be scrapped,
    # instead using hardcoded E, D and M matrices, which are much more friendly for the user
    @classmethod
    def load_from_files(
            cls,
            preprocessing: bool = True,
            **kwargs: dict[str, str | Path | pd.DataFrame] | None
        ) -> Self:
        valid_metrics = list(config.FILE_SELECTION_COUNTRY_DATA.keys())
        invalid_input_keys = set(kwargs.keys()) - set(valid_metrics)
        if invalid_input_keys:
            raise ValueError(
                f"Invalid metric(s) provided: {list(invalid_input_keys)}. "
                f"Select one or more of the following: {valid_metrics}"
            )
        if not kwargs:
            raise ValueError(
                f"At least one metric matrix must be provided. Options: {valid_metrics}"
            )

        dataset_instance = cls(country_code="CUSTOM")
        for metric in valid_metrics:
            setattr(dataset_instance, metric, None)
            if metric in kwargs and kwargs[metric] is not None:
                new_demo_grid = DemographicGridLoader.manual_load_from_file(
                    kwargs[metric], 
                    preprocessing
                )
                setattr(dataset_instance, metric, new_demo_grid)
        return dataset_instance

    def _initialize_and_validate(self) -> None:
        """Initializes the HMD data fetcher in the case that it was not initialized
        during an earlier load_data run. Also validates the country code and sets
        individual metrics to None.
        """
        if hasattr(self, "_data_fetcher"):
            return

        for metric in config.FILE_SELECTION_COUNTRY_DATA.keys():
            setattr(self, metric, None)

        fetcher = DataFetcherHMD(self.country_code)

        if not fetcher.is_country_code_valid():
            raise ValueError(f"Selected country code '{self.country_code}' is invalid")
        self._data_fetcher = fetcher