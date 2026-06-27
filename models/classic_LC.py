import numpy as np
import xarray as xr
from typing import Optional
from core.data_structures import MortalityData
from core.base_model import Model

class LeeCarterModel(Model):
    # TODO: add simulations to the __init__
    # TODO: add .x_ to parameters estimated by the model to comply with sklearn standards
    # TODO: completely redesign the model utilizing the sklearn's BaseEstimator
    def __init__(self, mortality_dataclass: MortalityData, value_column: str) -> None:
        """Initialize the Lee-Carter model with mortality data and target variable.

        Parameters
        ----------
        mortality_dataclass
            Dataclass containing the HMD mortality data.
        value_column
            The name of the column in the dataset for modeling (Male, Female or Total).
        """
        super().__init__(mortality_dataclass, value_column)
        self.ax: Optional[np.array] = None
        self.bx: Optional[np.array] = None
        self.kt: Optional[np.array] = None
        self.explained_variance: Optional[float] = None

        self.overlap_step = 0 if self.mortality_dataclass.overlap else 1


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
        self.drift = (self.kt[-1] - self.kt[0]) / (len(self.kt) - 1)
        self.std_of_errors = np.std(np.diff(self.kt) - self.drift)
        return self
    

    def predict_kt(self, steps: int, simulations: int) -> xr.DataArray:
        """Simulate and predict the stochastic walk of the kt parameter.
        """
        innovations = np.random.normal(
            self.drift, self.std_of_errors, size=(steps, simulations)
        )
        innovations = np.insert(innovations, 0, 0, axis=0)
        kt_forecast = self.kt[-1] + np.cumsum(innovations, axis=0)

        last_year = self.mortality_dataclass.year_interval["end"]
        pred_years = np.arange(self.overlap_step, steps + 1) + last_year

        return xr.DataArray(
            kt_forecast[self.overlap_step:],
            coords=[pred_years, np.arange(1, simulations + 1)],
            dims=["Year", "Simulation"],
            name="kt_forecast"
        )

    
    def predict_kt_analytical(self, steps: int) -> xr.DataArray:
        """Predict the values of the kt parameter analytically.
        """
        drifts = np.repeat(self.drift, steps)
        drifts = np.insert(drifts, 0, 0)
        kt_forecast = self.kt[-1] + np.cumsum(drifts, axis=0)

        last_year = self.mortality_dataclass.year_interval["end"]
        pred_years = np.arange(self.overlap_step, steps + 1) + last_year

        return xr.DataArray(
            kt_forecast[self.overlap_step:],
            coords=[pred_years],
            dims=["Year"],
            name="kt_forecast_analytical"
        )


    def predict(self, steps: int, simulations: int, stochastic: bool = True) -> xr.DataArray:
        """Predict the future mortality values.
        """
        if not stochastic and simulations > 1:
            simulations_nonstochastic = "A count of '1' has to be selected " \
            "for simulations in the case of an analytical prediction."
            raise ValueError(simulations_nonstochastic)

        if stochastic:
            kt_preds = self.predict_kt(steps, simulations)
            output_data_name = "mx_forecast"
        else:
            kt_preds = self.predict_kt_analytical(steps)
            output_data_name = "mx_forecast_analytical"

        ages = np.arange(len(self.ax))
        log_mx_preds = xr.DataArray(self.ax, dims="Age") + \
            xr.DataArray(self.bx, dims="Age") * kt_preds
        log_mx_preds = log_mx_preds.assign_coords(Age=ages)
        return np.exp(log_mx_preds).rename(output_data_name)

        
    # TODO: this should be completely removed and the user should compute it themselves
    def predict_historical(self):
        log_mx_matrix = self.ax[:, np.newaxis] + self.bx[:, np.newaxis] * self.kt
        
        return xr.DataArray(
            np.exp(log_mx_matrix),
            coords=[self.wide_matrix.to_pandas().index.to_numpy(), self.wide_matrix.to_pandas().columns.to_numpy()],
            dims=["Age", "Year"],
            name="mx_historical"
        )