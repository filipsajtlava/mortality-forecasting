import numpy as np
import xarray as xr
from typing import Optional
from core.data_structures import MortalityData
from core.base_model import Model

class LeeCarterModel(Model):
    def __init__(self, mortality_dataclass: MortalityData, value_column: str) -> None:
        """_summary_

        Parameters
        ----------
        mortality_dataclass
            _description_
        value_column
            _description_
        """
        super().__init__(mortality_dataclass, value_column)
        self.ax: Optional[np.array] = None
        self.bx: Optional[np.array] = None
        self.kt: Optional[np.array] = None
        self.explained_variance: Optional[float] = None

    def fit(self) -> "LeeCarterModel":
        """Fit the Lee-Carter model using SVD.
        """
    
        self.wide_matrix = self.wide_matrix.where(self.wide_matrix != 0, 1e-9)
        log_values = np.log(self.wide_matrix)
        ax_da = log_values.mean(axis=1)
        self.ax = ax_da.values
        centered_wide_matrix = log_values - ax_da

        U, s, V = np.linalg.svd(centered_wide_matrix.values, full_matrices=False)
        
        scaling_factor = U[:, 0].sum()
        self.bx = U[:, 0] / scaling_factor
        self.kt = s[0] * V[0, :] * scaling_factor

        self.explained_variance = s[0]**2 / np.sum(s**2)

        return self

    def predict(self, steps: int, simulations: int = 1) -> xr.DataArray:
        """Predict the future mortality values using a stochastic 
        projection of the kt parameter as a random walk with drift.
        """

        drift = (self.kt[-1] - self.kt[0]) / (len(self.kt) - 1)
        std_of_errors = np.std(np.diff(self.kt) - drift) # Random error

        last_year = self.mortality_dataclass.year_interval["end"]
        pred_years = np.arange(1, steps + 1) + last_year
        ages = np.arange(len(self.ax))

        if simulations == 1:
            random_walk = np.random.normal(0, scale=std_of_errors, size=steps) + drift
            kt_pred = np.cumsum(random_walk) + self.kt[-1]  # Shape: (steps,)
            
            log_mx_matrix = self.ax[:, np.newaxis] + self.bx[:, np.newaxis] * kt_pred
            
            last_year = self.mortality_dataclass.year_interval["end"]
            return xr.DataArray(
                np.exp(log_mx_matrix),
                coords=[ages, pred_years],
                dims=["Age", "Year"],
                name="MortalityRates"
            )
        
        else:
            random_walk = np.random.normal(0, scale=std_of_errors, size=(simulations, steps)) + drift
            kt_pred = (np.cumsum(random_walk, axis=1) + self.kt[-1]).T  # Shape: (steps, simulations)
            
            log_mx_tensor = self.ax[:, np.newaxis, np.newaxis] + self.bx[:, np.newaxis, np.newaxis] * kt_pred[np.newaxis, :, :]            
            return xr.DataArray(
                np.exp(log_mx_tensor),
                coords=[ages, pred_years, np.arange(1, simulations + 1)],
                dims=["Age", "Year", "Simulation"],
                name="MortalityRates"
            )
        
    

