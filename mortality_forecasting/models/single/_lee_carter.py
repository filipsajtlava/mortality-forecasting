from typing import Literal, Self

import numpy as np
import xarray as xr

from mortality_forecasting.models.single._base_single import SinglePopulationModel
from mortality_forecasting.core._commons import ParameterContainer, assert_glm
from mortality_forecasting import config
from mortality_forecasting.models.single._parameters import LeeCarterParams


class LeeCarterModel(SinglePopulationModel):
    _MODEL_PARAMS = LeeCarterParams()

    def __init__(
            self,
            method: Literal["SVD", "poisson", "negative_binomial"],
            num_initialization: Literal["naive", "SVD"] | ParameterContainer = "SVD",
            lee_miller_fix: bool = False,
            ftol: float = 1e-7,
            verbose: bool = False
        ) -> None:
        super().__init__(method=method, lee_miller_fix=lee_miller_fix)
        self.num_initialization = num_initialization
        self.ftol = ftol
        self.verbose = verbose

    @property
    def parameters_(self):
        return self._parameters

    # +=======================================+ #
    #               FITTING LOGIC               #
    # +=======================================+ #

    def _fit(self) -> None:
        method_map = {
            "SVD": self._fit_svd,
            "poisson": self._fit_glm,
            "negative_binomial": self._fit_glm
        }
        if self.method not in method_map:
            raise ValueError(f"The selected method '{self.method}' is unavailable.")

        ax_, bx_, kt_, *lambda_ = method_map[self.method]()
        self._parameters = self._MODEL_PARAMS._build_container({
            self._MODEL_PARAMS.AX: ax_,
            self._MODEL_PARAMS.BX: bx_,
            self._MODEL_PARAMS.KT: kt_
        })
        if lambda_:
            self._parameters.static["lambda_dispersion"] = lambda_[0] 
        #attrs={
        #        "overlap": self.mortality_data.D.overlap,
        #        "last_year": self.mortality_data.D.year_interval["end"]
        #    }

    def _fit_svd(self) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        if self.lee_miller_fix:
            last_year = self.D.coords[config.YEAR_DIM].max()
            override_M = getattr(self, "_jumpoff_rate_override", None)
            if override_M is not None:
                M_last_column = override_M.sel({config.YEAR_DIM: last_year})
            else:
                M_last_column = self.M.sel({config.YEAR_DIM: last_year})
            ax = np.log(M_last_column).drop_vars(config.YEAR_DIM)
        else:
            ax = np.log(self.M).mean(dim=config.YEAR_DIM)
        Z_centered = np.log(self.M) - ax

        bx, kt = xr.apply_ufunc(
            self._svd_kernel,
            Z_centered,
            input_core_dims=[[config.AGE_DIM, config.YEAR_DIM]],
            output_core_dims=[[config.AGE_DIM], [config.YEAR_DIM]],
            vectorize=True,
            output_dtypes=[float, float]
        )
        return ax, bx, kt

    @staticmethod
    def _svd_kernel(centered_matrix: np.ndarray):
        U, s, V = np.linalg.svd(centered_matrix, full_matrices=False)
        scaling_factor = U[:, 0].sum()
        bx = U[:, 0] / scaling_factor
        kt = s[0] * V[0, :] * scaling_factor
        return bx, kt

    def _initialize_parameters(self) -> dict[str, xr.DataArray | None]:
        initialized_params = {}
        lambda_init = None

        if self.num_initialization == "SVD":
            lc_model = LeeCarterModel(
                method=self.num_initialization
            ).fit(self.mortality_data)
            for parameter, da in lc_model.parameters_._non_scalars():
                initialized_params[parameter] = da
        elif self.num_initialization == "naive":
            for param_spec in self._MODEL_PARAMS:
                dims = [param_spec.dimension] + list(self._bound_dims.keys())
                initialized_params[param_spec.name] = xr.DataArray(
                    param_spec.naive_value, 
                    dims=dims,
                    coords={
                        **self._bound_dims,
                        param_spec.dimension: self.D.coords[param_spec.dimension].values
                    }
                )
        else:
            for param_spec in self._MODEL_PARAMS:
                initialized_params[param_spec.name] = self.num_initialization.get(
                    param_spec.name, None
                )
            lambda_init = self.num_initialization.get("lambda_dispersion", None)
            
        initialized_params["lambda_dispersion"] = self.family.dispersion_initialization(
            bound_dims=self._bound_dims,
            init=lambda_init
        )
        return initialized_params

    def _compute_deaths(
            self, 
            ax: xr.DataArray, 
            bx: xr.DataArray, 
            kt: xr.DataArray
        ) -> xr.DataArray:
        return self.E * np.exp(ax + bx * kt)

    def _fit_glm(
            self
        ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray | None]:
        param_init = self._initialize_parameters()
        ax, bx, kt = (
            param_init[self._MODEL_PARAMS.AX.name], 
            param_init[self._MODEL_PARAMS.BX.name],
            param_init[self._MODEL_PARAMS.KT.name]
        )
        lambda_dispersion = param_init.get("lambda_dispersion", None)
        self.ll_history_ = [
            self.family.compute_log_likelihood(
                D=self.D,
                E=self.E,
                D_pred=self._compute_deaths(ax, bx, kt),
                lambda_dispersion=lambda_dispersion
            )
        ]
        slots_active_mask = xr.full_like(self.ll_history_[0], fill_value=True, dtype=bool)
        iteration = 0

        while bool(slots_active_mask.any()) and iteration <= config.MAXIMUM_LC_ITERATIONS:
            iteration += 1

            D_pred = self._compute_deaths(ax, bx, kt)
            num_factor, denom_factor = self.family.numerical_optimization_factor(
                D=self.D,
                D_pred=D_pred,
                lambda_dispersion=lambda_dispersion
            )
            # tends to extremely overshoot for the naive initialization, the clipping
            # makes sure it stays in 'normal' values. In the SVD initialization, there's basically
            # no problem
            ax_candidate = (
                ax +
                ((self.D - num_factor * D_pred).sum(dim=config.YEAR_DIM) / 
                (denom_factor * D_pred).sum(dim=config.YEAR_DIM)).clip(min=-10, max=10)
            )
            ax = xr.where(slots_active_mask, ax_candidate, ax)

            D_pred = self._compute_deaths(ax, bx, kt)
            num_factor, denom_factor = self.family.numerical_optimization_factor(
                D=self.D,
                D_pred=D_pred,
                lambda_dispersion=lambda_dispersion
            )
            bx_candidate = (
                bx + (kt * (self.D - num_factor * D_pred)).sum(dim=config.YEAR_DIM) / 
                (denom_factor * D_pred * kt*kt).sum(dim=config.YEAR_DIM)
            )
            bx_candidate_sum = bx_candidate.sum(dim=config.AGE_DIM)
            bx_candidate = bx_candidate / bx_candidate_sum
            kt_candidate = kt * bx_candidate_sum
            bx = xr.where(slots_active_mask, bx_candidate, bx)
            kt = xr.where(slots_active_mask, kt_candidate, kt)

            D_pred = self._compute_deaths(ax, bx, kt)
            num_factor, denom_factor = self.family.numerical_optimization_factor(
                D=self.D,
                D_pred=D_pred,
                lambda_dispersion=lambda_dispersion
            )
            kt_candidate = (
                kt + (bx * (self.D - num_factor * D_pred)).sum(dim=config.AGE_DIM) / 
                (denom_factor * D_pred * bx*bx).sum(dim=config.AGE_DIM)
            )
            kt_candidate_mean = kt_candidate.mean(dim=config.YEAR_DIM)
            kt_candidate = kt_candidate - kt_candidate_mean
            ax_candidate = ax + bx * kt_candidate_mean
            ax = xr.where(slots_active_mask, ax_candidate, ax)
            kt = xr.where(slots_active_mask, kt_candidate, kt)

            D_pred = self._compute_deaths(ax, bx, kt)
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
            last_year = self.D.coords[config.YEAR_DIM].max()
            override_M = getattr(self, "_jumpoff_rate_override", None)
            if override_M is not None:
                M_last_column = override_M.sel({config.YEAR_DIM: last_year})
            else:
                M_last_column = self.M.sel({config.YEAR_DIM: last_year})
            ax = (
                np.log(M_last_column) - bx * kt.sel({config.YEAR_DIM: last_year})
            ).drop_vars(config.YEAR_DIM)

        return (
            (ax, bx, kt) if (lambda_dispersion == None).any()
            else (ax, bx, kt, lambda_dispersion)
        )

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