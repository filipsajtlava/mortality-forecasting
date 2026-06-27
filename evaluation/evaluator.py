import xarray as xr
import numpy as np
from core.data_structures import MortalityData
from core.base_model import Model


class MortalityEvaluator:
    def __init__(self, model_fitted: Model, testing_data: MortalityData, simulations: int):
        """Initialize the evaluator to compare model projections against testing data.

        Parameters
        ----------
        model_fitted
            The fitted mortality model instance.
        testing_data
            The observed data used for validation.
        simulations 
            Number of stochastic trajectories to generate.

        Raises
        ------
        ValueError
            If the model has not been fitted or is missing required parameters.
        """
        self.model = model_fitted
        self.testing_data = testing_data
        self.simulations = simulations
        self.value_column = self.model.value_column

        # TODO: THIS NEEDS REDESIGNING FOR OTHER MODELS WHEN THEY ARE GOING TO USE SOMETHING OTHER THAN ax
        if not hasattr(self.model, "ax") or self.model.ax is None:
            raise ValueError("You need to fit the model first.")

        self.predicted_data = self._calculate_predictions()


    def _calculate_predictions(self) -> None:
        """Calculates the predictions for the period of the testing data using the fitted model.
        """
        self.years_to_predict = (
            self.testing_data.year_interval["end"] - \
            self.model.mortality_dataclass.year_interval["end"]
        )
        return self.model.predict(steps=self.years_to_predict, simulations=self.simulations)


    def _calculate_MAE(self) -> float:
        """Calculates the MAE of aggregated predictions and the test set

        Returns
        -------
            MAE error.
        """
        abs_percent_errors = np.abs(
            self.testing_data.get_pivoted_data(self.value_column) - self.agg_predictions
        )
        
        return float(abs_percent_errors.mean())


    def _calculate_RMSE(self) -> float:
        """Calculates the RMSE of aggregated predictions and the test set

        Returns
        -------
            RMSE error.
        """
        squared_errors = (
            np.log(self.agg_predictions) - \
            np.log(self.testing_data.get_pivoted_data(self.value_column))
        ) ** 2

        return float(np.sqrt(squared_errors.mean()))    


    def _calculate_MASE(self) -> float:
        """Calculates the MASE of aggregated predictions and the test set

        Returns
        -------
            MASE error.
        """
        abs_error = np.abs(self.testing_data.get_pivoted_data(self.value_column) - self.agg_predictions)
        training_set = self.model.wide_matrix
    
        return abs_error
        return float(abs_error.mean() / np.abs(training_set.diff(dim="Year")).mean())


    def calculate_error(
            self, 
            aggregate: str = "mean", 
            error: str = None, 
            start_year: int = None
        ) -> xr.DataArray:
        """Compute the difference between observed and predicted mortality values.

        Parameters
        ----------
        method, optional
            Aggregation method for simulations ("mean" or "median"), by default "mean".

        Returns
        -------
            The specified error.

        Raises
        ------
        ValueError
            Incorrectly specified method for aggregating or incorrectly specified error.
        """
        aggregate_methods = ["mean", "median"]
        
        error_methods = {
            "MAE": self._calculate_MAE,
            "RMSE": self._calculate_RMSE,
            "MASE": self._calculate_MASE
        }

        if aggregate in aggregate_methods:
            if aggregate == "mean":
                self.agg_predictions = self.predicted_data.mean(dim="Simulation")
            elif aggregate == "median":
                self.agg_predictions = self.predicted_data.median(dim="Simulation")
        else:
            error_message =  f"Selected method '{aggregate}' isn't allowed. Choose one from {aggregate_methods}"
            raise ValueError(error_message)

        if start_year is not None:
            self.agg_predictions = self.agg_predictions.sel(Year=slice(start_year, None))

        if error in error_methods.keys():
            return error_methods[error]()
        else:
            error_message = f"Selected error '{error}' is not allowed. Choose one from {error_methods}"
            raise ValueError(error_message)
        
