from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
import xarray as xr

from mortality_forecasting import config
from mortality_forecasting.data_processing._hmd_data_fetcher import DataFetcherHMD


class MortalityDataset:
    @classmethod
    def load_from_hmd(
            cls,
            country_codes: str | Sequence[str],
            starting_year: int,
            ending_year: int,
            maximum_age: int = 90,
        ) -> xr.Dataset:
        countries = [country_codes] if isinstance(country_codes, str) else list(country_codes)
        country_datasets = []

        for country in countries:
            fetcher = DataFetcherHMD(country)
            if not fetcher.is_country_code_valid():
                raise ValueError(f"Invalid country code: '{country}'")

            loaded_paths = fetcher.fetch_country_data()
            metric_arrays = {}

            for metric in config.FILE_SELECTION_COUNTRY_DATA.keys():
                if metric in loaded_paths:
                    da = cls._parse_hmd_file(
                        file_path=loaded_paths[metric],
                        country_code=country,
                        starting_year=starting_year,
                        ending_year=ending_year,
                        maximum_age=maximum_age,
                    )
                    metric_arrays[metric] = da

            if metric_arrays:
                country_datasets.append(xr.Dataset(metric_arrays))

        if not country_datasets:
            raise ValueError("No data loaded for the provided country codes.")

        mortality_dataset = xr.concat(country_datasets, dim=config.COUNTRY_DIM, join="outer")
        if isinstance(country_codes, str):
            mortality_dataset = mortality_dataset.squeeze(config.COUNTRY_DIM, drop=True)
        return mortality_dataset

    @classmethod
    def load_from_files(
            cls,
            input_dict: dict[str, str | Path | pd.DataFrame | xr.DataArray],
            country_code: str = "CUSTOM",
            preprocessing: bool = True,
        ) -> xr.Dataset:
        metric_arrays = {}

        for metric, raw_input in input_dict.items():
            if isinstance(raw_input, (str, Path)):
                df = pd.read_csv(raw_input, index_col=0)
            elif isinstance(raw_input, pd.DataFrame):
                df = raw_input
            elif isinstance(raw_input, xr.DataArray):
                metric_arrays[metric] = raw_input
                continue
            else:
                raise TypeError(f"Invalid input type for metric '{metric}': {type(raw_input)}")

            df.index = df.index.astype(int)
            df.columns = df.columns.astype(int)
            da = df.stack(future_stack=True).rename_axis([config.AGE_DIM, config.YEAR_DIM]).to_xarray()

            if config.COUNTRY_DIM not in da.dims:
                da = da.expand_dims({config.COUNTRY_DIM: [country_code]})
            if config.SEX_DIM not in da.dims:
                da = da.expand_dims({config.SEX_DIM: [config.SEX[2]]})

            if preprocessing:
                da = cls._preprocess_array(da)

            metric_arrays[metric] = da

        return xr.Dataset(metric_arrays)

    @classmethod
    def train_test_split(
            cls,
            ds: xr.Dataset,
            split_year: int,
            overlap: bool = False
        ) -> tuple[xr.Dataset, xr.Dataset]:
        min_year = int(ds.coords[config.YEAR_DIM].min())
        max_year = int(ds.coords[config.YEAR_DIM].max())

        if not (min_year <= split_year <= max_year):
            raise ValueError(f"Split year {split_year} out of bounds ({min_year}-{max_year}).")

        test_start = split_year if overlap else split_year + 1

        train = ds.sel({config.YEAR_DIM: slice(min_year, split_year)})
        test = ds.sel({config.YEAR_DIM: slice(test_start, max_year)})

        return train, test

    @classmethod
    def _parse_hmd_file(
            cls,
            file_path: Path,
            country_code: str,
            starting_year: int,
            ending_year: int,
            maximum_age: int,
        ) -> xr.DataArray:
        raw_df = pd.read_csv(file_path, sep=r"\s+", header=1, na_values=".")

        raw_df[config.AGE_DIM] = (
            raw_df[config.AGE_DIM]
            .astype(str)
            .str.replace("+", "", regex=False)
            .astype(int)
        )

        query_str = (
            f"{config.YEAR_DIM} >= {starting_year} and "
            f"{config.YEAR_DIM} <= {ending_year} and "
            f"{config.AGE_DIM} <= {maximum_age}"
        )
        filtered_df = raw_df.query(query_str)

        sex_cols = [c for c in ["Female", "Male", "Total"] if c in filtered_df.columns]
        df_long = (
            filtered_df.set_index([config.AGE_DIM, config.YEAR_DIM])[sex_cols]
            .stack()
            .reset_index()
        )
        df_long.columns = [config.AGE_DIM, config.YEAR_DIM, config.SEX_DIM, "value"]

        da = df_long.set_index([config.AGE_DIM, config.YEAR_DIM, config.SEX_DIM])["value"].to_xarray()
        da = da.expand_dims({config.COUNTRY_DIM: [country_code]})

        return cls._preprocess_array(da)

    @staticmethod
    def _preprocess_array(da: xr.DataArray) -> xr.DataArray:
        return np.exp(
            np.log(da.where(da > 0, 1e-9))
            .interpolate_na(dim=config.AGE_DIM, method="linear", fill_value="extrapolate")
        ).transpose(*config.DIM_ORDERING, missing_dims="ignore")