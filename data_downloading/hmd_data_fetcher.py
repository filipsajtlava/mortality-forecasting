import requests
import io
import re
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from config import (
    STALE_FALLBACK_COUNTRY_AVAILABLE,
    CACHED_COUNTRY_AVAILABLE,
    CACHED_TTL_COUNTRY_CODE_DAYS,
    CACHED_TTL_COUNTRY_DATA_DAYS,
    FILE_SELECTION_COUNTRY_DATA,
)

class DataFetcherHMD:
    def __init__(self, data_parent_directory_path: Path, country_code: str) -> None:
        self._data_parent_directory_path = data_parent_directory_path
        self._country_code = country_code

        self._username = os.getenv("HMD_USERNAME")
        self._password = os.getenv("HMD_PASSWORD")
        self._session = None
        
        self._data_parent_directory_path.mkdir(exist_ok=True)


    def is_country_code_valid(self) -> bool:
        """Checks whether the stored country code exists in the HMD's list of available countries.

        Uses a cached list of valid country codes when available and not expired,
        refetching from the HMD website otherwise. Falls back to a bundled stale
        list if the refetch fails.

        Returns
        -------
            True if the country code is valid, False otherwise.
        """
        cache_codes_path = self._data_parent_directory_path / CACHED_COUNTRY_AVAILABLE
        fetch_error_code = None

        if not cache_codes_path.exists() or \
        self._is_cache_expired(cache_codes_path, CACHED_TTL_COUNTRY_CODE_DAYS):
            fetch_error_code = self._fetch_valid_country_codes(cache_codes_path)

        if fetch_error_code == 200 or fetch_error_code is None:
            comparison_source_path = cache_codes_path
        else:
            error_msg = ("Defaulting to a fallback list")
            print(error_msg)
            comparison_source_path = Path(__file__).parent / STALE_FALLBACK_COUNTRY_AVAILABLE
            
        with comparison_source_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return True if data.get(self._country_code) else False


    def _fetch_valid_country_codes(self, path_to_cache) -> int:
        """Fetches the current list of valid HMD country codes and caches it locally as JSON.

        Parameters
        ----------
        path_to_cache
            Path to the local file where the fetched country code list should be cached.

        Returns
        -------
            HTTP status code of the fetch request.
        """
        session = requests.Session()
        response = session.get(
            "https://www.mortality.org/File/GetDocument/Public/country_codes.csv",
            timeout=30
        )
        if response.status_code == 200:
            country_code_df = pd.read_csv(io.StringIO(response.text))
            code_country_paired = dict(zip(country_code_df.iloc[:, 1], country_code_df.iloc[:, 0]))
            with open(path_to_cache, "w", encoding="utf-8") as f:
                json.dump(code_country_paired, f, indent=4)

        return response.status_code


    def fetch_country_data(self) -> dict[str, Path]:
        """Loads cached datasets or downloads data files for the stored country code from the HMD.

        Checks for existing, non-expired cached files for each configured data type
        (e.g. mx, ex, dx) and downloads any that are missing or stale, authenticating
        a session if credentials are required.

        Returns
        -------
            Mapping of data type keys (e.g. "mx", "ex", "dx") to the local file paths
            that were successfully retrieved or already cached.
        """
        country_data_path = self._data_parent_directory_path / self._country_code
        country_data_path.mkdir(exist_ok=True)
    
        failed_downloads = []
        successfuly_loaded = {}
        for key, file in FILE_SELECTION_COUNTRY_DATA.items():
            path_to_file = country_data_path / file
            
            if not path_to_file.exists() or \
            self._is_cache_expired(path_to_file, CACHED_TTL_COUNTRY_DATA_DAYS):
                self._ensure_credentials_present()
                if self._session is None:
                    self._initialize_session()
                download_error = self._download_data(file, path_to_file)
                if download_error:
                    failed_downloads.append(file)
                else:
                    successfuly_loaded[key] = path_to_file
            else:
                successfuly_loaded[key] = path_to_file
        if not any(country_data_path.iterdir()):
            country_data_path.rmdir()

        if failed_downloads:
            print("There has been a problem with fetching the data of these specific combinations:\n")
            print("COUNTRY CODE               FILE")
            print("===============================")
            for failed_file in failed_downloads:
                print(f"{self._country_code:<10} {failed_file:>20}")
        
        return successfuly_loaded


    def _ensure_credentials_present(self) -> None:
        """Checks if credentials are present in the .env file.
        """
        if not self._password or not self._username:
            missing_credentials_error = (
                "\nThis script uses the official datasets provided by the Human Mortality Database (HMD): \n\n" 
                "https://www.mortality.org/\n\n"\
                "Please create an account and enter your login credentials into the '.env' file, as described in the 'setup' section. \n" \
                "This message is shown because this instance could not locate your credentials.\n"
            )
            raise ValueError(missing_credentials_error)


    def _initialize_session(self) -> None:
        """Initializes and authenticates a network session with the Human Mortality Database.

        Raises
        ------
        ConnectionError
            If the security token could not be found on the login page.
        """
        session = requests.Session() # Create a session so the web 'remembers' us
        login_url = "https://www.mortality.org/Account/Login"
        response = session.get(login_url, timeout=30)  
        token_match = re.search(r'name="__RequestVerificationToken" type="hidden" value="([^"]+)"', response.text) # Look for verification token
        if not token_match:
            raise ConnectionError("Could not find security token.")
        session.post(login_url, data={ # Login into HMD's web
            "Email": self._username,
            "Password": self._password,
            "__RequestVerificationToken": token_match.group(1),
            "ReturnUrl": ""
        })
        self._session = session


    def _download_data(self, file: str, path_to_file: Path) -> bool:
        """Downloads a specific HMD file for the stored country code.

        Parameters
        ----------
        file
            The file name to download from the HMD (e.g. "Mx_1x1.txt").
        path_to_file
            Local path where the downloaded file should be saved.

        Returns
        -------
            True if a download error occurred, False otherwise.
        """
        data_url = f"https://www.mortality.org/File/GetDocument/hmd.v6/{self._country_code}/STATS/{file}"
        data_response = self._session.get(data_url, timeout=30)
        content_type = data_response.headers.get("Content-type", "")
        download_error = "text/html" in content_type or data_response.status_code != 200
        if not download_error:
            with open(path_to_file, mode="w", encoding="UTF-8") as f:
                f.write(data_response.text)

        return download_error


    def _is_cache_expired(self, path_to_cache: Path, expiration_ttl_days) -> bool:
        """Checks whether a cached file is older than the allowed time-to-live.

        Parameters
        ----------
        path_to_cache
            Path to the cached file whose age should be checked.
        expiration_ttl_days
            Maximum allowed age of the cached file, in days, before it's considered stale.

        Returns
        -------
            True if the file's age meets or exceeds the TTL, False otherwise.
        """
        file_modified_time = datetime.fromtimestamp(
            path_to_cache.stat().st_mtime,
            tz=timezone.utc
        )
        age_of_file = datetime.now(timezone.utc) - file_modified_time

        return age_of_file.days >= expiration_ttl_days