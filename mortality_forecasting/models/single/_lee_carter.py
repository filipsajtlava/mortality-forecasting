from typing import Literal, Self

import numpy as np
import xarray as xr

from mortality_forecasting.models.single._base_single import SinglePopulationModel
from mortality_forecasting.core._commons import ParameterContainer, assert_glm
from mortality_forecasting import config


class LeeCarterModel(SinglePopulationModel):
    def __init__(
            self,
            method: Literal["SVD", "poisson", "negative_binomial"],
            num_initialization: Literal["naive", "SVD"] | ParameterContainer = "SVD",
            lee_miller_fix: bool = False,
            bootstrap_anchor: Literal["partial", "full"] = "partial", # TODO: this is only used by the
            # bootstrapper to give it the information about the anchor type, that cant be the proper 
            # approach of doing this, id suggest having a boostrapper instance
            ftol: float = 1e-7,
            verbose: bool = False
        ) -> None:
        super().__init__(method=method, lee_miller_fix=lee_miller_fix)
        self.num_initialization = num_initialization
        self.ftol = ftol
        self.verbose = verbose
        self.bootstrap_anchor = bootstrap_anchor

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

        self._parameters = ParameterContainer(
            static=xr.Dataset(
                data_vars={
                    "ax": ax_,
                    "bx": bx_
                }
            ),
            period=xr.Dataset(
                data_vars={
                    "kt": kt_
                },                
                attrs={
                    "overlap": self.mortality_data.D.overlap,
                    "last_year": self.mortality_data.D.year_interval["end"]
                }
            )
        )
        if lambda_:
            self._parameters.static["lambda_dispersion"] = lambda_[0]

    def _fit_svd(self) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        if self.lee_miller_fix:
            last_year = self.mortality_data.D.year_interval["end"]
            override_M = getattr(self, "_jumpoff_rate_override", None)
            if override_M is not None:
                M_last_column = override_M.sel({config.YEAR_DIM: last_year})
            else:
                M_last_column = self.M.sel({config.YEAR_DIM: last_year})
            ax = np.log(M_last_column).drop_vars(config.YEAR_DIM)
        else:
            ax = np.log(self.M).mean(dim=config.YEAR_DIM)
        Z_centered = np.log(self.M) - ax

        U, s, V = np.linalg.svd(Z_centered.values, full_matrices=False)
        
        scaling_factor = U[:, 0].sum()
        bx = xr.DataArray(
            U[:, 0] / scaling_factor, 
            coords=[(config.AGE_DIM, np.log(self.M)[config.AGE_DIM].values)]
        )
        kt = xr.DataArray(
            s[0] * V[0, :] * scaling_factor, 
            coords=[(config.YEAR_DIM, np.log(self.M)[config.YEAR_DIM].values)]
        )

        self.explained_variance_ = s[0]**2 / np.sum(s**2)
        return ax, bx, kt

    def _initialize_parameters(
            self
        ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, float | None]:
        ages = self.D[config.AGE_DIM].values
        years = self.D[config.YEAR_DIM].values

        if self.num_initialization == "SVD":
            lc_model = LeeCarterModel(
                method=self.num_initialization
            ).fit(self.mortality_data, self.value_column)
            ax = lc_model.parameters_["ax"]
            bx = lc_model.parameters_["bx"]
            kt = lc_model.parameters_["kt"]
            lambda_dispersion = self.family.dispersion_initialization()
        elif self.num_initialization == "naive":
            ax = xr.DataArray(0, coords=[(config.AGE_DIM, ages)])
            bx = xr.DataArray(0, coords=[(config.AGE_DIM, ages)])
            kt = xr.DataArray(1, coords=[(config.YEAR_DIM, years)])
            lambda_dispersion = self.family.dispersion_initialization()
        else:
            ax = self.num_initialization.get("ax", None)
            bx = self.num_initialization.get("bx", None)
            kt = self.num_initialization.get("kt", None)
            lambda_init = self.num_initialization.get("lambda_dispersion", None)
            lambda_dispersion = self.family.dispersion_initialization(
                init=float(lambda_init) if lambda_init is not None else None
            )
        return ax, bx, kt, lambda_dispersion

    def _compute_deaths(
            self, 
            ax: xr.DataArray, 
            bx: xr.DataArray, 
            kt: xr.DataArray
        ) -> xr.DataArray:
        return self.E * np.exp(ax + bx * kt)

    def _fit_glm(
            self
        ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, float | None]:
        ax, bx, kt, lambda_dispersion = self._initialize_parameters()
        self.ll_history_ = [
            self.family.compute_log_likelihood(
                D=self.D,
                E=self.E,
                D_pred=self._compute_deaths(ax, bx, kt),
                lambda_dispersion=lambda_dispersion
            )
        ]
        likelihood_change = np.inf
        iteration = 0

        while (
            likelihood_change > self.ftol and 
            iteration <= config.MAXIMUM_LC_ITERATIONS
        ):
            iteration += 1

            D_pred = self._compute_deaths(ax, bx, kt)
            num_factor, denom_factor = self.family.numerical_optimization_factor(
                D=self.D,
                D_pred=D_pred,
                lambda_dispersion=lambda_dispersion
            )
            ax_new = (
                ax + (self.D - num_factor * D_pred).sum(dim=config.YEAR_DIM) / 
                (denom_factor * D_pred).sum(dim=config.YEAR_DIM)
            )

            # Each iteration takes new parameters
            D_pred = self._compute_deaths(ax_new, bx, kt)
            num_factor, denom_factor = self.family.numerical_optimization_factor(
                D=self.D,
                D_pred=D_pred,
                lambda_dispersion=lambda_dispersion
            )
            bx_new = (
                bx + (kt * (self.D - num_factor * D_pred)).sum(dim=config.YEAR_DIM) / 
                (denom_factor * D_pred * kt*kt).sum(dim=config.YEAR_DIM)
            )
            bx_sum = bx_new.sum()
            bx_new = bx_new / bx_sum
            kt = kt * bx_sum

            D_pred = self._compute_deaths(ax_new, bx_new, kt)
            num_factor, denom_factor = self.family.numerical_optimization_factor(
                D=self.D,
                D_pred=D_pred,
                lambda_dispersion=lambda_dispersion
            )
            kt_new = (
                kt + (bx_new * (self.D - num_factor * D_pred)).sum(dim=config.AGE_DIM) / 
                (denom_factor * D_pred * bx_new*bx_new).sum(dim=config.AGE_DIM)
            )
            kt_mean = kt_new.mean()
            kt_new = kt_new - kt_mean
            ax_new = ax_new + bx_new * kt_mean

            D_pred = self._compute_deaths(ax_new, bx_new, kt_new)
            lambda_dispersion_new = self.family.update_dispersion(
                D=self.D,
                D_pred=D_pred,
                lambda_dispersion=lambda_dispersion
            )
            self.ll_history_.append(
                self.family.compute_log_likelihood(
                    D=self.D,
                    E=self.E,
                    D_pred=D_pred,
                    lambda_dispersion=lambda_dispersion_new
                )
            )
            prev_ll = self.ll_history_[-2]
            curr_ll = self.ll_history_[-1]
            if abs(prev_ll) < 1e-10:
                likelihood_change = abs(curr_ll - prev_ll)
            else:
                likelihood_change = abs((curr_ll - prev_ll) / prev_ll)
            if self.verbose:
                print(f"iteration {iteration}: log-change: {likelihood_change}")

            ax = ax_new
            bx = bx_new
            kt = kt_new
            lambda_dispersion = lambda_dispersion_new

        if iteration > config.MAXIMUM_LC_ITERATIONS:
            print(
                f"WARNING: the maximum amount of iterations " \
                f"({config.MAXIMUM_LC_ITERATIONS}) has been reached. " \
                f"The algorithm might not have converged."
            )

        if self.lee_miller_fix:
            last_year = self.mortality_data.D.year_interval["end"]
            override_M = getattr(self, "_jumpoff_rate_override", None)
            if override_M is not None:
                M_last_column = override_M.sel({config.YEAR_DIM: last_year})
            else:
                M_last_column = self.M.sel({config.YEAR_DIM: last_year})
            ax = (
                np.log(M_last_column) - bx * kt.sel({config.YEAR_DIM: last_year})
            ).drop_vars(config.YEAR_DIM)

        return (ax, bx, kt) if lambda_dispersion is None else (ax, bx, kt, lambda_dispersion)

    def _predict_mortalities(
            self, 
            forecasted_values: ParameterContainer
        ) -> xr.DataArray:
        log_M_predictions = (
            forecasted_values.static.ax + 
            forecasted_values.static.bx * forecasted_values.period.kt
        )
        return np.exp(log_M_predictions)

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