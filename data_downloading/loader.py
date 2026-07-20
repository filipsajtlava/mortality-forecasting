from data_downloading.hmd_data_fetcher import DataFetcherHMD
from pathlib import Path
import config
from core.data_structures import MortalityData
import pandas as pd

class CountryData:
    def __init__(self, country_code: str) -> None:
        self.mx: MortalityData | None = None
        self.ex: MortalityData | None = None
        self.dx: MortalityData | None = None

        self.country_code = country_code
        self._country_code_valid = False
        hmd_data_path = Path.cwd() / config.DATA_DIRECTORY_NAME 
        self._data_fetcher = DataFetcherHMD(hmd_data_path, country_code)


    def load_data(self, data_type: str, starting_year: int, ending_year: int, maximum_age: int = 90) -> None:
        """Loads cached datasets or downloads data from the HMD database filtering by the specified arguments.
        
        Parameters
        ----------
        data_type
            The mortality quantity to load: "mx" (central death rates), "ex" (exposures) or "dx" (deaths)
        starting_year
            First year included in the time frame.
        ending_year
            Last year included in the time frame.
        maximum_age, optional
            Maximum age included in the dataset, by default 90.
        """
        if not (self._country_code_valid or self._data_fetcher.is_country_code_valid()):
            raise ValueError("Selected country code is invalid")

        self._country_code_valid = True
        successfuly_loaded = self._data_fetcher.fetch_country_data()
        if data_type in successfuly_loaded:
            setattr(self, data_type, self._minor_preprocessing(successfuly_loaded[data_type], starting_year, ending_year, maximum_age))
        else:
            raise ValueError(f"The combination {self.country_code}-{data_type} could not be retrieved.")


    def _minor_preprocessing(self, full_path: str, starting_year: int, ending_year: int, maximum_age: int) -> MortalityData:
        """Minimal preprocessing of the downloaded HMD files.

        Returns
        -------
            MortalityData instance holding all of the data and the additional information together in one place
        """
        data = pd.read_csv(full_path, sep=r"\s+", header=1, na_values=".")
        data["Age"] = data["Age"].astype(str).str.replace("+", "", regex=False).astype(int) # We need to remove the "+" from 110+ to be able to use filters
        data = data.query(f"Year >= {starting_year} and Year <= {ending_year} and Age <= {maximum_age}")
        # TODO: This has to be redesigned with log values, to account for Gompertzs law
        pivoted_values = data.pivot(
            index="Year", 
            columns="Age", 
            values=["Female", "Male", "Total"]
        ).interpolate(method="linear", axis=0, limit_direction="both") 
        interpolated_data = pivoted_values.stack(level="Age").reset_index()

        return MortalityData(interpolated_data)