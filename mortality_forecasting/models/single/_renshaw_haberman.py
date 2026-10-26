from typing import Literal, Self

import numpy as np
import xarray as xr

from mortality_forecasting.models.single._base_single import SinglePopulationModel
from mortality_forecasting.core._commons import ParameterContainer, assert_glm
from mortality_forecasting import config
from mortality_forecasting.models.single._parameters import RenshawHabermanParams, ParamSpecification


class RenshawHabermanModel(SinglePopulationModel):
    _PARAMS = RenshawHabermanParams()

    def __init__(
            self,
            method: Literal["poisson", "negative_binomial"],
            num_initialization: Literal["naive", "SVD"] | ParameterContainer = "SVD",
            lee_miller_fix: bool = False,
            model_type: Literal["M", "H1"] = "M",
            ftol: float = 1e-7,
            cohort_count_minimum: int = 5,
            verbose: bool = False
        ) -> None:
        super().__init__(method=method, lee_miller_fix=lee_miller_fix)
        self.model_type = model_type
        self.num_initialization = num_initialization
        self.ftol = ftol
        self.cohort_count_minimum = cohort_count_minimum
        self.verbose = verbose

    @property
    def parameters_(self):
        return self._parameters

    def _fit(self) -> None:
        method_map = {
            "poisson": self._fit_glm,
            "negative_binomial": self._fit_glm
        }
        if self.method not in method_map:
            raise ValueError(f"The selected method '{self.method}' is unavailable.")
        return method_map[self.method]()

    def _compute_deaths(
            self, 
            ax: xr.DataArray, 
            bx1: xr.DataArray, 
            kt: xr.DataArray,
            bx2: xr.DataArray,
            gc: xr.DataArray
        ) -> xr.DataArray:
        gc = gc.sel({config.COHORT_DIM: self._year_coords - self._age_coords})
        return self.E * np.exp(ax + bx1 * kt + bx2 * gc)

    def _fit_glm(
            self
        ) -> dict[ParamSpecification, xr.DataArray]:
        cohort_matrix = (
            self._year_coords - self._age_coords
        ).to_pandas().stack().value_counts()
        valid_cohorts = np.sort(
            cohort_matrix[cohort_matrix >= self.cohort_count_minimum]
            .index
            .values
        )
        self._weights = (
            self._year_coords - self._age_coords
        ).isin(valid_cohorts).astype("float")

        param_init = self._initialize_parameters()
        ax, bx1, kt, bx2, gc = (
            param_init[self._PARAMS.AX.name], 
            param_init[self._PARAMS.BX1.name],
            param_init[self._PARAMS.KT.name],
            param_init[self._PARAMS.BX2.name],
            param_init[self._PARAMS.GC.name]
        )
        if self.model_type == "H1":
            bx2 = xr.DataArray(
                1 / self.D.sizes[config.AGE_DIM],
                dims=[self._PARAMS.BX2.dimension] + list(self._bound_dims.keys()),
                coords={
                    **self._bound_dims,
                    self._PARAMS.BX2.dimension: (
                        np.unique(self._year_coords - self._age_coords)
                        if self._PARAMS.BX2.dimension == config.COHORT_DIM 
                        else self.D.coords[self._PARAMS.BX2.dimension].values
                    )
                }
            )
        gc = gc.where(gc.coords[config.COHORT_DIM].isin(valid_cohorts), 0.0)

        lambda_dispersion = param_init.get("lambda_dispersion", None)
        self.ll_history_ = [
            self.family.compute_log_likelihood(
                D=self.D,
                E=self.E,
                D_pred=self._compute_deaths(ax, bx1, kt, bx2, gc),
                lambda_dispersion=lambda_dispersion
            )
        ]
        slots_active_mask = xr.full_like(self.ll_history_[0], fill_value=True, dtype=bool)
        iteration = 0

        while bool(slots_active_mask.any()) and iteration <= config.MAXIMUM_LC_ITERATIONS:
            iteration += 1

            D_pred = self._compute_deaths(ax, bx1, kt, bx2, gc)
            num_factor, denom_factor = self.family.numerical_optimization_factor(
                D=self.D,
                D_pred=D_pred,
                lambda_dispersion=lambda_dispersion
            )
            ax_candidate = ax + (
                (
                    (self._weights * (self.D - num_factor * D_pred)).sum(dim=config.YEAR_DIM) 
                    / 
                    (self._weights * denom_factor * D_pred).sum(dim=config.YEAR_DIM)
                )
                .fillna(0.0)
                .clip(min=-10, max=10)
            )
            ax = xr.where(slots_active_mask, ax_candidate, ax)

            D_pred = self._compute_deaths(ax, bx1, kt, bx2, gc)
            num_factor, denom_factor = self.family.numerical_optimization_factor(
                D=self.D,
                D_pred=D_pred,
                lambda_dispersion=lambda_dispersion
            )
            bx1_candidate = bx1 + (
                (
                    (self._weights * kt * (self.D - num_factor * D_pred)).sum(dim=config.YEAR_DIM) 
                    / 
                    (self._weights * denom_factor * D_pred * kt*kt).sum(dim=config.YEAR_DIM)
                )
                .fillna(0.0)
                .clip(min=-10, max=10)
            )
            bx1_candidate_sum = bx1_candidate.sum(dim=config.AGE_DIM)
            bx1_candidate = bx1_candidate / bx1_candidate_sum
            kt_candidate = kt * bx1_candidate_sum
            bx1 = xr.where(slots_active_mask, bx1_candidate, bx1)
            kt = xr.where(slots_active_mask, kt_candidate, kt)

            D_pred = self._compute_deaths(ax, bx1, kt, bx2, gc)
            num_factor, denom_factor = self.family.numerical_optimization_factor(
                D=self.D,
                D_pred=D_pred,
                lambda_dispersion=lambda_dispersion
            )
            kt_candidate = kt + (
                (
                    (self._weights * bx1 * (self.D - num_factor * D_pred)).sum(dim=config.AGE_DIM) 
                    / 
                    (self._weights * denom_factor * D_pred * bx1*bx1).sum(dim=config.AGE_DIM)
                )
                .fillna(0.0)
                .clip(min=-10, max=10)
            )
            kt_candidate_mean = kt_candidate.mean(dim=config.YEAR_DIM)
            kt_candidate = kt_candidate - kt_candidate_mean
            ax_candidate = ax + bx1 * kt_candidate_mean
            ax = xr.where(slots_active_mask, ax_candidate, ax)
            kt = xr.where(slots_active_mask, kt_candidate, kt)

            if self.model_type == "M": 
                D_pred = self._compute_deaths(ax, bx1, kt, bx2, gc)
                num_factor, denom_factor = self.family.numerical_optimization_factor(
                    D=self.D,
                    D_pred=D_pred,
                    lambda_dispersion=lambda_dispersion
                )
                bx2_candidate = bx2 + (
                    (
                        (
                            self._weights * 
                            gc.sel({config.COHORT_DIM: self._year_coords - self._age_coords}) * 
                            (self.D - num_factor * D_pred)
                        )
                        .sum(dim=config.YEAR_DIM) 
                        / 
                        (
                            self._weights * 
                            denom_factor * D_pred * 
                            gc.sel({config.COHORT_DIM: self._year_coords - self._age_coords}) ** 2
                        )
                        .sum(dim=config.YEAR_DIM)
                    )
                    .fillna(0.0)
                    .clip(min=-10, max=10)
                )
                bx2_candidate_sum = bx2_candidate.sum(dim=config.AGE_DIM)
                bx2_candidate = bx2_candidate / bx2_candidate_sum
                gc_candidate = gc * bx2_candidate_sum
                bx2 = xr.where(slots_active_mask, bx2_candidate, bx2)
                gc = xr.where(slots_active_mask, gc_candidate, gc)

            D_pred = self._compute_deaths(ax, bx1, kt, bx2, gc)
            num_factor, denom_factor = self.family.numerical_optimization_factor(
                D=self.D,
                D_pred=D_pred,
                lambda_dispersion=lambda_dispersion
            )
            gc_candidate = gc + (
                (
                    (self._weights * bx2 * (self.D - num_factor * D_pred))
                    .assign_coords(Cohort=self._year_coords - self._age_coords)
                    .groupby(config.COHORT_DIM).sum() 
                    / 
                    (self._weights * denom_factor * bx2*bx2 * D_pred)
                    .assign_coords(Cohort=self._year_coords - self._age_coords)
                    .groupby(config.COHORT_DIM).sum()
                )
                .fillna(0.0)
                .clip(min=-10, max=10)
                .sel(Cohort=gc.coords[config.COHORT_DIM])
            )
            # Constraint one
            gc_candidate_mean = gc_candidate.mean(dim=config.COHORT_DIM)
            gc_candidate = gc_candidate - gc_candidate_mean
            gc_candidate = gc_candidate.where(
                gc.coords[config.COHORT_DIM].isin(valid_cohorts), 0.0
            )
            ax_candidate = ax + bx2 * gc_candidate_mean

            # Constraint two
            y_coords = gc_candidate.coords[config.COHORT_DIM]
            y_centered = y_coords - y_coords.mean()
            trend_slope = (
                (gc_candidate * y_centered).sum(dim=config.COHORT_DIM) 
                / 
                (y_centered ** 2).sum(dim=config.COHORT_DIM)
            )
            gc_candidate = gc_candidate - trend_slope * y_centered
            gc_candidate_mean = gc_candidate.mean(dim=config.COHORT_DIM)
            gc_candidate = gc_candidate - gc_candidate_mean
            gc_candidate = gc_candidate.where(
                gc.coords[config.COHORT_DIM].isin(valid_cohorts), 0.0
            )
            ax_candidate = ax_candidate + bx2 * gc_candidate_mean
            ax = xr.where(slots_active_mask, ax_candidate, ax)
            gc = xr.where(slots_active_mask, gc_candidate, gc)
            print((gc * y_centered).sum())

            D_pred = self._compute_deaths(ax, bx1, kt, bx2, gc)
            lambda_dispersion_candidate = self.family.update_dispersion(
                D=self.D,
                D_pred=D_pred,
                lambda_dispersion=lambda_dispersion
            )
            lambda_dispersion = xr.where(
                slots_active_mask, 
                lambda_dispersion_candidate, 
                lambda_dispersion
            )

            self.ll_history_.append(
                self.family.compute_log_likelihood(
                    D=self.D,
                    E=self.E,
                    D_pred=D_pred,
                    lambda_dispersion=lambda_dispersion
                )
            )
            prev_ll = self.ll_history_[-2]
            curr_ll = self.ll_history_[-1]
            likelihood_change = xr.where(
                abs(prev_ll) < 1e-10, 
                abs(curr_ll - prev_ll), 
                abs((curr_ll - prev_ll) / prev_ll)
            )
            slots_active_mask = likelihood_change > self.ftol

            if self.verbose:
                print(f"iteration {iteration}: log-change: {likelihood_change}")

        if iteration > config.MAXIMUM_LC_ITERATIONS:
            print(
                f"WARNING: the maximum amount of iterations " \
                f"({config.MAXIMUM_LC_ITERATIONS}) has been reached. " \
                f"The algorithm might not have converged."
            )

        if self.lee_miller_fix:
            pass
            #last_year = self.D.coords[config.YEAR_DIM].max()
            #override_M = getattr(self, "_jumpoff_rate_override", None)
            #if override_M is not None:
            #    M_last_column = override_M.sel({config.YEAR_DIM: last_year})
            #else:
            #    M_last_column = self.M.sel({config.YEAR_DIM: last_year})
            #ax = (
            #    np.log(M_last_column) - bx * kt.sel({config.YEAR_DIM: last_year})
            #).drop_vars(config.YEAR_DIM)

        return {
            self._PARAMS.AX: ax,
            self._PARAMS.BX1: bx1,
            self._PARAMS.KT: kt,
            self._PARAMS.BX2: bx2,
            self._PARAMS.GC: gc
        }

    def _predict_mortalities(
            self, 
            forecasted_values: ParameterContainer
        ) -> xr.DataArray:
        log_M_predictions = (
            forecasted_values.static.ax + 
            forecasted_values.static.bx * forecasted_values.period.kt
        )
        return np.exp(log_M_predictions).transpose(*config.DIM_ORDERING, missing_dims="ignore")

    # TODO: if other models will use the lee-miller correction, this should be moved into the parent class 
    def _set_jumpoff_anchor(self, rate: xr.DataArray | None) -> Self:
        self._jumpoff_rate_override = rate
        return self

    # +=======================================+ #
    #                 RESIDUALS                 #
    # +=======================================+ #

    @property
    def deviance_residuals(self) -> xr.DataArray:
        assert_glm(self)
        return self.family.get_deviance_residuals(
            D=self.D,
            D_pred=self.predict_in_sample() * self.E,
            lambda_dispersion=self.parameters_.get("lambda_dispersion")
        )

    @property
    def pearson_residuals(self) -> xr.DataArray:
        assert_glm(self)
        D_pred = self.predict_in_sample() * self.E
        variance = self.family.get_variance(
            D_pred=D_pred,
            lambda_dispersion=self.parameters_.get("lambda_dispersion")
        )
        return (self.D - D_pred) / np.sqrt(variance)