from __future__ import annotations
from typing import Literal, TYPE_CHECKING

import xarray as xr

from mortality_forecasting.core._base_forecaster import Forecaster
from mortality_forecasting.data_processing._dataset import MortalityDataset
from mortality_forecasting import config
from mortality_forecasting.core._commons import (
    bounds_from_simulations, 
    ForecastContainer, 
    ParameterContainer,
    assert_glm
)
if TYPE_CHECKING:
    from mortality_forecasting.core._base_model import Model



class Bootstrapper:
    def __init__(
            self,
            bootstrap_anchor: Literal["partial", "full"] = "partial",
            return_residuals: Literal["deviance", "pearson"] | None = None
        ) -> None:
        self.bootstrap_anchor = bootstrap_anchor
        self.return_residuals = return_residuals

    # TODO: this wont run for dual forecaster
    def run(
            self, 
            model: Model, 
            forecaster: Forecaster, 
            steps: int,
            alpha: float
        ) -> ForecastContainer:
        assert_glm(model)
        self._assert_simulations_present(forecaster)
        value_column = model.value_column
        param_forecasts = []
        mortality_forecasts = []

        in_sample_residuals = [] if self.return_residuals is not None else None
        residual_method = f"{self.return_residuals}_residuals"

        parameters = model.parameters_
        overlap=parameters.period.attrs["overlap"]
        D_pred = model.predict_in_sample() * model.E
        jump_off_anchor = model.M if self.bootstrap_anchor == "full" else None

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

            if self.return_residuals is not None:
                residuals = getattr(new_model, residual_method)
                in_sample_residuals.append(residuals)

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
            in_sample_residuals,
            forecaster,
            alpha
        )

    def _concatenate_outputs(
            self,
            forecasted_parameters: list[ParameterContainer],
            forecasted_mortalities: list[xr.DataArray],
            residuals: list[xr.DataArray] | None,
            forecaster: Forecaster,
            alpha: float
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
                        concat_params, alpha, forecaster.point_estimate
                    )
                param_types[param_type] = concat_params
        
        concat_mortalities = xr.concat(
            forecasted_mortalities,
            dim=config.SIMULATION_DIM,
            data_vars="all"
        ).assign_coords({config.SIMULATION_DIM: sim_coords})
        if residuals is not None:
            concat_residuals = xr.concat(
                residuals,
                dim=config.SIMULATION_DIM,
                data_vars="all"
            ).assign_coords({config.SIMULATION_DIM: sim_coords})
        else:
            concat_residuals = None

        if not forecaster.return_simulations:
            concat_mortalities = bounds_from_simulations(
                concat_mortalities, alpha, forecaster.point_estimate
            )
            concat_residuals = bounds_from_simulations(
                concat_residuals, alpha, forecaster.point_estimate
            )

        return ForecastContainer(
            mortality_rates_=concat_mortalities,
            parameters_=ParameterContainer(
                static=param_types["static"],
                period=param_types["period"],
                cohort=param_types["cohort"]
            ),
            in_sample_residuals_=concat_residuals,
            attrs={"alpha": alpha}
        )

    def _assert_simulations_present(self, forecaster: Forecaster):
        if getattr(forecaster, "simulations", None) is None:
            raise ValueError(
                "To use bootstrapping, please enter simulations into the forecaster."
            )