from pathlib import Path

import pandas as pd
import numpy as np


def generate_synthetic_dataset(
        start_age: int,
        end_age: int,
        start_year: int,
        end_year: int,
        yearly_decrease: float,
        exposure_values: float,
        alpha_makeham: float,
        beta_makeham: float
    ) -> None:
    ages = np.arange(start_age, end_age + 1)
    years = np.arange(start_year, end_year + 1)
    shape = (len(ages), len(years))
    time_factor = yearly_decrease ** years

    exp_synthetic = pd.DataFrame(
        np.full(shape, exposure_values), index=ages, columns=years
    )

    # Makeham's mortality law
    mortality_synthetic = pd.DataFrame(
        np.outer(alpha_makeham * (beta_makeham ** ages), time_factor), 
        index=ages, 
        columns=years
    )

    death_synthetic = exp_synthetic * mortality_synthetic

    save_path = Path(__file__).parents[1] / "data"
    exp_synthetic.to_csv(save_path / "SYNTHETIC_Exposures.csv")
    mortality_synthetic.to_csv(save_path / "SYNTHETIC_Mortalities.csv")
    death_synthetic.to_csv(save_path / "SYNTHETIC_Deaths.csv")

if __name__ == "__main__":
    generate_synthetic_dataset(
        start_age=0, 
        end_age=100, 
        start_year=1950, 
        end_year=2010,
        yearly_decrease=0.995,
        exposure_values=896.852,
        alpha_makeham=0.00005,
        beta_makeham=1.08
    )
