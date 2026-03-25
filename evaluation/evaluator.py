import xarray as xr
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
        years_to_predict = self.testing_data.year_interval["end"] - self.model.mortality_dataclass.year_interval["end"]
        return self.model.predict(steps=years_to_predict, simulations=self.simulations)

    def calculate_residuals(self, method: str = "mean") -> xr.DataArray:
        """Compute the difference between observed and predicted mortality values.

        Parameters
        ----------
        method, optional
            Aggregation method for simulations ("mean" or "median"), by default "mean".

        Returns
        -------
        xr.DataArray
            The calculated residuals (Observed - Predicted).

        Raises
        ------
        ValueError
            If the specified method is not in ['mean', 'median'].
        """
        possible_methods = ["mean", "median"]
    
        if method not in possible_methods:
            raise ValueError(f"Selected method '{method}' isn't allowed. Choose one from {possible_methods}")
        else:
            if method == "mean":
                collapsed_predicted_data = self.predicted_data.mean(dim="Simulation")
            elif method == "median":
                collapsed_predicted_data = self.predicted_data.median(dim="Simulation")

        residuals = self.testing_data.get_pivoted_data(self.value_column) - collapsed_predicted_data
        return residuals