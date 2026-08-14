from pathlib import Path

import pandas as pd
import numpy as np

from data_downloading.datasets import DemographicGrid


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
        target_columns = ["Female", "Male", "Total"]
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