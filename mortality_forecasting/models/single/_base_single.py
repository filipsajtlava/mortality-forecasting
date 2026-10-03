from abc import ABC, abstractmethod
from typing import Self, Literal

from mortality_forecasting.core._base_model import Model
from mortality_forecasting.data_processing._dataset import MortalityDataset
from mortality_forecasting import config
from mortality_forecasting.core._commons import validate_value_column, ForecastContainer, ParameterContainer
from mortality_forecasting.core._base_forecaster import Forecaster
from mortality_forecasting.forecasting._dual_forecaster import DualForecaster
from mortality_forecasting.forecasting._bootstrapper import Bootstrapper


class SinglePopulationModel(Model, ABC):
    @abstractmethod
    def _fit(self):
        pass

    def fit(self, mortality_data: MortalityDataset, value_column: str) -> Self:
        """Fit the model on the mortality data with a specified value column,
        using an individually set up method.

        Parameters
        ----------
        mortality_data
            Instance of the MortalityDataset, with loaded data depending on
            the model architecture, similar to 'x' in sklearn.
        value_column
            The chosen value column used for fitting the model,
            similar to 'y' in sklearn.
        """
        validate_value_column(value_column)
        self._validate_dataset(mortality_data, value_column)

        self.mortality_data = mortality_data
        self.value_column = value_column
        self.D = self.mortality_data.D[self.value_column]
        self.E = self.mortality_data.E[self.value_column]
        self.M = self.D / self.E

        self._fit()
        return self

    def forecast(
            self, 
            forecaster: Forecaster | DualForecaster,
            steps: int,
            alpha: float = 0.05,
            bootstrap: bool = False
        ) -> ForecastContainer:
        self._check_if_fitted()

        if bootstrap is True:
            bootstrapper = Bootstrapper(alpha=alpha)
            return bootstrapper.run(model=self, forecaster=forecaster, steps=steps)

        if isinstance(forecaster, Forecaster):
            forecaster.fit(self.parameters_.period)
            period_ds = forecaster.forecast_parameters(steps, alpha)
            parameters = ParameterContainer(
                static=self.parameters_.static,
                period=period_ds
            )
        elif isinstance(forecaster, DualForecaster):
            forecaster.fit(self.parameters_, alpha)
            period_ds, cohort_ds = forecaster.forecast_parameters(steps, alpha)
            parameters = ParameterContainer(
                static=self.parameters_.static,
                period=period_ds,
                cohort=cohort_ds
            )
        else:
            raise ValueError("Please enter a valid forecaster instance.")

        predicted_mortalities = self._predict_mortalities(parameters)
        return ForecastContainer(
            mortality_rates_=predicted_mortalities,
            parameters_=parameters, 
            attrs={"alpha": alpha}
        )

    def _validate_dataset(
            self,
            mortality_data: MortalityDataset,
            value_column: str
        ) -> None:
        """Check if the specified datasets are present in the MortalityDataset
        instance (only exposures and deaths). 
        
        If there is more than one required grid to be checked, this method also
        validates that every grid contains the exact same timespan.

        Parameters
        ----------
        mortality_data
            Instance of the MortalityDataset.
        value_column
            Specified value column that has to appear in the dataset.
        """
        reference_grid = None
        for grid in config.REQUIRED_DATA_FOR_MODELS:
            selected_grid = getattr(mortality_data, grid, None)
            if selected_grid is None:
                raise ValueError(
                    f"Dataset does not contain the grid '{grid}'."
                )
            else:
                try:
                    selected_grid[value_column]
                except:
                    raise ValueError(
                        "The selected column is not available in the dataset."
                    )

                # TODO: this year-interval mismatch checker could be moved to commons,
                # and maybe used in the manual loader, to look if the years are the same
                if reference_grid is None:
                    reference_grid = selected_grid

                if reference_grid.year_interval != getattr(mortality_data, grid).year_interval:
                    raise ValueError(
                        f"Year interval mismatch between grid " \
                        f"'{reference_grid}' and grid '{grid}'."
                    )