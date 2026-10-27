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
            input_dict: (
                dict[str, str | Path | xr.DataArray]
                | dict[str, dict[str, str | Path | xr.DataArray]]
            ),
            preprocessing: bool = True,
        ) -> xr.Dataset:
        countries = ["CUSTOM"]
        country_datasets = []

        first_val = list(input_dict.values())[0]
        if isinstance(first_val, dict):
            countries = list(input_dict.keys())
        else:
            input_dict = {countries[0]: input_dict}

        for country in countries:
            metric_arrays = {}
            for metric, raw_input in input_dict[country].items():
                da = cls._parse_custom_input(
                    raw_input=raw_input,
                    metric=metric,
                    country_code=country,
                    preprocessing=preprocessing,
                )
                metric_arrays[metric] = da

            if metric_arrays:
                country_datasets.append(xr.Dataset(metric_arrays))

        if not country_datasets:
            raise ValueError("No data loaded from the provided inputs.")

        if len(country_datasets) == 1:
            return country_datasets[0]
        else:
            return xr.concat(
                country_datasets, 
                dim=config.COUNTRY_DIM, 
                join="outer"
            )

    @classmethod
    def _parse_custom_input(
            cls,
            raw_input: str | Path | pd.DataFrame | xr.DataArray,
            metric: str,
            country_code: str,
            preprocessing: bool = True,
        ) -> xr.DataArray:
        if isinstance(raw_input, (str, Path)):
            df = pd.read_csv(raw_input, index_col=0)
            df.index = df.index.astype(int)
            df.columns = df.columns.astype(int)
            da = (
                df
                .stack(future_stack=True)
                .rename_axis([config.AGE_DIM, config.YEAR_DIM])
                .to_xarray()
            )
        elif isinstance(raw_input, xr.DataArray):
            da = raw_input.copy()
        else:
            raise TypeError(f"Invalid input type for metric '{metric}': {type(raw_input)}")

        if config.COUNTRY_DIM not in da.dims and country_code != "CUSTOM":
            da = da.expand_dims({config.COUNTRY_DIM: [country_code]})
        if config.SEX_DIM not in da.dims:
            da = da.expand_dims({config.SEX_DIM: [config.SEX[2]]})

        if preprocessing:
            da = cls._preprocess_array(da)
        return da

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