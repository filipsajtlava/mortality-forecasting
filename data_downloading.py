import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import re
from typing import Optional
from config import plot_configuration
from core.data_structures import MortalityData

class CountryData:
    def __init__(self, country_hmd_code: str) -> None:
        self.mx: Optional[MortalityData] = None
        self.ex: Optional[MortalityData] = None
        self.dx: Optional[MortalityData] = None
        self.data_directory = "hmd_data"

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
        path = os.path.join(self.data_directory, self.country_code)
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
                self._check_credentials_present()
                if session is None:
                    session = self._initialize_session()
                download_error = self._download_data(file, path_to_file, session)

            if download_error:
                failed_downloads.append(file)
            else:
                setattr(self, key, self._minor_preprocessing(path_to_file, starting_year, ending_year, maximum_age))

        if len(os.listdir(path)) == 0:
            os.rmdir(path)

        if failed_downloads:
            print("There has been a problem with fetching the data of these specific combinations:\n")
            print("COUNTRY CODE               FILE")
            print("===============================")
            for failed_file in failed_downloads:
                print(f"{self.country_code:<10} {failed_file:>20}")


    def _check_credentials_present(self) -> None:
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


    def _initialize_session(self) -> requests.Session:
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


    def _download_data(self, file: str, path_to_file: str, session: requests.Session) -> bool:
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


    def _minor_preprocessing(self, full_path: str, starting_year: int, ending_year: int, maximum_age: int) -> MortalityData:
        """Minimal preprocessing of the downloaded HMD files.

        Returns
        -------
            MortalityData instance holding all of the data and the additional information together in one place
        """
        data = pd.read_csv(full_path, sep=r"\s+", header=1, na_values=".")
        data["Age"] = data["Age"].astype(str).str.replace("+", "", regex=False).astype(int) # We need to remove the "+" from 110+ to be able to use filters
        data = data.query(f"Year >= {starting_year} and Year <= {ending_year} and Age <= {maximum_age}")
        pivoted_values = data.pivot(index="Year", 
                                    columns="Age", 
                                    values=["Female", "Male", "Total"]
        ).interpolate(method="linear", axis=0, limit_direction="both") 
        interpolated_data = pivoted_values.stack(level="Age").reset_index()

        return MortalityData(interpolated_data)
    
        # TODO: MOVE INTO DATA STRUCTURES - MORTALITYDATA (PROBABLY)
    def plot_age_profiles(self, year_step: int = 10, legend_size: float = 10, value_column: str = "Total") -> plt.Axes:
        """Plot the development of mx based on Age and Year.

        Parameters
        ----------
        year_step, optional
            Size of individuals steps in the sequence between the first and the last year, by default 10.
        legend_size, optional
            Legend size multiplicator, the higher the number, the bigger the legend, by default 10.
        value_column, optional
            Column of values we use for visualization, by default "Total".

        Returns
        -------
            Matplotlib axis object containing the plot.

        Raises
        ------
        ValueError
            Empty MortalityData.mx object.
        """
        if self.mx is None:
            raise ValueError("The mx data is not loaded.")
        
        year_range = np.arange(self.mx.year_interval["start"], self.mx.year_interval["end"] + 1, year_step)
        plotting_dataframe = (
            self.mx.data.loc[
                self.mx.data.Year.isin(year_range), 
                ["Age", "Year", value_column]
            ]
            .pivot(index="Age", columns="Year", values=value_column)
        )

        with plt.style.context("seaborn-v0_8"):
            y_lower_bound, y_upper_bound = (plotting_dataframe.max().max() * -0.02, plotting_dataframe.max().max() * 0.5)
            x_lower_bound, x_upper_bound = (max(60, plotting_dataframe.index.min()), plotting_dataframe.index.max())
            ax = plotting_dataframe.plot(cmap="magma")

            limit_age = 90
            ax.axvline(limit_age, color="black", linestyle="dashed", linewidth=1)

            ax.set_xlim([x_lower_bound, x_upper_bound])
            ax.set_ylim([y_lower_bound, y_upper_bound])
            ax.set_ylabel(f"Death rates ({value_column})")
            ax.set_title("Age profiles throughout the years")
            plot_configuration(ax, legend_location="upper left", legend_size=legend_size)
            return ax


