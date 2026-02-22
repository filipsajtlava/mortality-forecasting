import pandas as pd
import os
import requests
import re
from dataclasses import dataclass

DATA_DIRECTORY_NAME = "hmd_data"

@dataclass # Used solely for making navigation in the imported data and its history easier
class MortalityDataclass:
    data: pd.DataFrame
    missing_values: dict[str, int]
    year_interval: dict[str, int]

class CountryData:
    def __init__(self, country_hmd_code: str) -> None:
        self.mx = None
        self.ex = None
        self.dx = None

        self.country_code = country_hmd_code

        self.username = os.getenv("HMD_USERNAME")
        self.password = os.getenv("HMD_PASSWORD")


    def load_data(self, starting_year: int, ending_year: int, maximum_age: int = 90) -> None:
        """ Loads cached datasets or downloads data from the HMD database filtering by the specified arguments.
        
        Parameters
        ----------
        starting_year
            First year included in the time frame.
        ending_year
            Last year included in the time frame.
        maximum_age, optional
            Maximum age included in the dataset, by default 90.
        """
        path = os.path.join(DATA_DIRECTORY_NAME, self.country_code)
        os.makedirs(path, exist_ok=True)
        files = {
            "mx": "Mx_1x1.txt",
            "ex": "Exposures_1x1.txt",
            "dx": "Deaths_1x1.txt"
        }

        session = None
        failed_downloads = []

        for key, file in files.items():
            download_error = False
            path_to_file = os.path.join(path, file)
            if not os.path.exists(path_to_file):
                self.check_credentials_present()
                if session is None:
                    session = self.initialize_session()
                download_error = self.download_data(file, path_to_file, session)

            if download_error:
                failed_downloads.append(file)
            else:
                setattr(self, key, self.minor_preprocessing(path_to_file, starting_year, ending_year, maximum_age))

        if len(os.listdir(path)) == 0:
            os.rmdir(path)

        if failed_downloads:
            print("There has been a problem with fetching the data of these specific combinations:\n")
            print("COUNTRY CODE               FILE")
            print("===============================")
            for failed_file in failed_downloads:
                print(f"{self.country_code:<10} {failed_file:>20}")


    def check_credentials_present(self) -> None:
        """Checks if credentials are present in the .env file.
        """
        if not self.password or not self.username:
            missing_credentials_error = (
                "\nThis script uses the official datasets provided by the Human Mortality Database (HMD): \n\n" 
                "https://www.mortality.org/\n\n"\
                "Please create an account and enter your login credentials into the '.env' file, as described in the 'setup' section. \n" \
                "This message is shown because this instance could not locate your credentials.\n"
            )
            raise ValueError(missing_credentials_error)


    def initialize_session(self) -> requests.Session:
        """Initializes the session with the Human Mortality Database.

        Returns
        -------
            Authenticated session object.

        Raises
        ------
        ConnectionError
            There's a problem with creating the connection.
        """
        session = requests.Session() # Create a session so the web 'remembers' us
        login_url = "https://www.mortality.org/Account/Login"
        response = session.get(login_url)  
        token_match = re.search(r'name="__RequestVerificationToken" type="hidden" value="([^"]+)"', response.text) # Look for verification token
        if not token_match:
            raise ConnectionError("Could not find security token.")
        session.post(login_url, data={ # Login into HMD's web
            "Email": self.username,
            "Password": self.password,
            "__RequestVerificationToken": token_match.group(1),
            "ReturnUrl": ""
        })

        return session


    def download_data(self, file: str, path_to_file: str, session: requests.Session) -> bool:
        """Downloads a specific HMD file.

        Returns
        -------
            True if a download error occurred, False otherwise.
        """
        data_url = f"https://www.mortality.org/File/GetDocument/hmd.v6/{self.country_code}/STATS/{file}"
        data_response = session.get(data_url)
        download_error = "<!DOCTYPE html>" in data_response.text[0:50] or data_response.status_code != 200

        if not download_error:
            with open(path_to_file, mode="w", encoding="UTF-8") as f:
                f.write(data_response.text)

        return download_error


    def minor_preprocessing(self, full_path: str, starting_year: int, ending_year: int, maximum_age: int) -> MortalityDataclass:
        """Minimal preprocessing of the downloaded HMD files.

        Returns
        -------
            MortalityDataclass instance holding all of the data and the additional information together in one place
        """
        data = pd.read_csv(full_path, sep=r"\s+", header=1, na_values=".")
        data["Age"] = data["Age"].astype(str).str.replace("+", "", regex=False).astype(int) # We need to remove the "+" from 110+ to be able to use filters
        data = data.query(f"Year >= {starting_year} and Year <= {ending_year} and Age <= {maximum_age}")
        pivoted_values = data.pivot(index="Year", 
                                    columns="Age", 
                                    values=["Female", "Male", "Total"]
        ).interpolate(method="linear", axis=0, limit_direction="both") 
        interpolated_data = pivoted_values.stack(level="Age").reset_index()
        numerical_columns = ["Female", "Male", "Total"]
        interpolated_data[numerical_columns] = interpolated_data[numerical_columns].replace(0, 1e-9) # Adding a small epsilon to the zeros in the dataset so that we can use logarithms
        
        # Under we add additional historic information of the dataset in order to make the process more transparent
        total_values = data.size
        missing_values = int(data.isna().sum().sum())
        percentage_missing = round(missing_values / total_values, 3)
        data_missing_values = {
            "total": data.size,
            "missing_pre_interpolation": missing_values, # Total amount of empty values in the entire matrix/dataframe
            "percent": percentage_missing
        }
        year_interval = {
            "start": starting_year, 
            "end": ending_year
        }

        return MortalityDataclass(interpolated_data, data_missing_values, year_interval)