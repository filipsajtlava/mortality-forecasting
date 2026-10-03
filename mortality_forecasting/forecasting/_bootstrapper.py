from typing import Literal

import xarray as xr

from mortality_forecasting.core._base_forecaster import Forecaster
from mortality_forecasting.core._base_model import Model
from mortality_forecasting.data_processing._dataset import MortalityDataset
from mortality_forecasting import config
from mortality_forecasting.core._commons import (
    bounds_from_simulations, 
    ForecastContainer, 
    ParameterContainer,
    assert_glm
)


class Bootstrapper:
    def __init__(
            self, 
            alpha: float
        ) -> None:
        self.alpha = alpha

    # TODO: this wont run for dual forecaster
    def run(
            self, 
            model: Model, 
            forecaster: Forecaster, 
            steps: int,
        ) -> ForecastContainer:
        assert_glm(model)
        self._assert_simulations_present(forecaster)
        value_column = model.value_column
        param_forecasts = []
        mortality_forecasts = []

        parameters = model.parameters_
        overlap=parameters.period.attrs["overlap"]
        D_pred = model.predict_in_sample() * model.E
        jump_off_anchor = model.M if model.bootstrap_anchor == "full" else None

        for loop_rng in forecaster.seed.spawn(forecaster.simulations):
            rng_sample, rng_forecast = loop_rng.spawn(2)
            samples = model.family.sample_from_distribution(
                D_pred=D_pred,
                seed=rng_sample,
                lambda_dispersion=parameters.get("lambda_dispersion", None)
            )
            data = MortalityDataset.load_from_files(
                E = {value_column: model.E},
                D = {value_column: samples},
                overlap=overlap
            )
            new_model = type(model)(
                method=model.method, 
                lee_miller_fix=model.lee_miller_fix,
                num_initialization=parameters,
                ftol=model.ftol,
                verbose=model.verbose
            )._set_jumpoff_anchor(jump_off_anchor).fit(data, model.value_column)

            forecaster_single = type(forecaster)(
                seed=rng_forecast,
                simulations=1,
                return_simulations=True
            )
            forecast = new_model.forecast(forecaster_single, steps)
            param_forecasts.append(
                forecast.parameters_
            )
            mortality_forecasts.append(
                forecast.mortality_rates_
            )

        return self._concatenate_outputs(
            param_forecasts,
            mortality_forecasts,
            forecaster
        )

    def _concatenate_outputs(
            self,
            forecasted_parameters: list[ParameterContainer],
            forecasted_mortalities: list[xr.DataArray],
            forecaster: Forecaster
        ) -> ForecastContainer:
        param_types = {"static": None, "period": None, "cohort": None}
        sim_coords = range(1, forecaster.simulations + 1) 

        for param_type in param_types.keys():
            if getattr(forecasted_parameters[0], param_type) is not None:
                concat_params = xr.concat(
                    [getattr(ind_parameter, param_type) for ind_parameter in forecasted_parameters],
                    dim=config.SIMULATION_DIM, 
                    data_vars="all"
                ).assign_coords({config.SIMULATION_DIM: sim_coords})
                if not forecaster.return_simulations:
                    concat_params = bounds_from_simulations(
                        concat_params, self.alpha, forecaster.point_estimate
                    )
                param_types[param_type] = concat_params
        
        concat_mortalities = xr.concat(
            forecasted_mortalities,
            dim=config.SIMULATION_DIM,
            data_vars="all"
        ).assign_coords({config.SIMULATION_DIM: sim_coords})
        if not forecaster.return_simulations:
            concat_mortalities = bounds_from_simulations(
                concat_mortalities, self.alpha, forecaster.point_estimate
            )

        return ForecastContainer(
            mortality_rates_=concat_mortalities,
            parameters_=ParameterContainer(
                static=param_types["static"],
                period=param_types["period"],
                cohort=param_types["cohort"]
            ),
            attrs={"alpha": self.alpha}
        )

    def _assert_simulations_present(self, forecaster: Forecaster):
        if getattr(forecaster, "simulations", None) is None:
            raise ValueError(
                "To use bootstrapping, please enter simulations into the forecaster."
            )