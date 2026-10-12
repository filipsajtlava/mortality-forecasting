from __future__ import annotations
from typing import Any, Literal, TYPE_CHECKING
from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes

from mortality_forecasting import config
from mortality_forecasting.core._base_plotter import Plotter
from mortality_forecasting.core._commons import assert_glm
if TYPE_CHECKING:
    from mortality_forecasting.core._base_model import Model


class ModelPlotter(Plotter):
    def __init__(self, model: Model) -> None:
        self._model = model
        self._model._check_if_fitted()
        self._axs_needed = sum(len(value) for value in self._model._bound_dims.values())

    # TODO: this method needs a comprehensive docstring to explain the plot config
    def plot_parameters(
            self, 
            axs: Sequence[Axes] | None = None,
            ax_settings: dict[str, Any] = {},
            line_settings: dict[str, Any] = {}
        ) -> list[Axes]:
        params = list(self._model.parameters_._non_scalars())

        countries = self._model._bound_dims.get(config.COUNTRY_DIM, [None])
        sexes = self._model._bound_dims.get(config.SEX_DIM, [None])

        axs = self._validate_and_normalize_axs(
            axs, len(params) * len(sexes), ax_settings
        )

        i = 0
        for parameter, parameter_da in params:
            for sex in sexes:
                current_da = (
                    parameter_da.sel({config.SEX_DIM: sex}) 
                    if config.SEX_DIM in parameter_da.dims 
                    else parameter_da
                )
                for c_idx, country in enumerate(countries):
                    country_da = (
                        current_da.sel({config.COUNTRY_DIM: country}) 
                        if config.COUNTRY_DIM in current_da.dims and country is not None
                        else current_da
                    )
                    x_dim = country_da.dims[0]
                    x_axis = country_da.coords[x_dim]

                    line_defaults = {"label": country if country else None}
                    has_countries = countries[0] is not None
                    curr_line_settings = {
                        k: (v[c_idx] if isinstance(v, list) and has_countries else (v[i] if isinstance(v, list) else v)) 
                        for k, v in line_settings.items()
                    }

                    axs[i].plot(
                        x_axis, 
                        country_da.values, 
                        **(line_defaults | curr_line_settings)
                    )

                    ylabel = f"Parameter {parameter} ({sex})" if sex else f"Parameter {parameter}"
                    axs[i].set(xlabel=x_dim, ylabel=ylabel)
                if countries[0] is not None:
                    axs[i].legend()
                i += 1
        return axs

    def plot_mortality_insample_heatmap(
            self,
            axs: Sequence[Axes] | None = None,
            ax_settings: dict[str, Any] = {},
            colorbar_settings: dict[str, Any] = {},
            imshow_settings: dict[str, Any] = {}
        ) -> list[Axes]:
        countries = self._model._bound_dims.get(config.COUNTRY_DIM, [None])
        sexes = self._model._bound_dims.get(config.SEX_DIM, [None])

        total_plots = len(sexes) * len(countries)
        axs = self._validate_and_normalize_axs(
            axs, total_plots, ax_settings, colorbar_settings, imshow_settings
        )

        predicted = self._model.predict_in_sample()
        residuals = (self._model.M - predicted) / np.sqrt(predicted)

        colorbar_settings = {"label": "Standardized mortality residuals"} | colorbar_settings

        i = 0
        for sex in sexes:
            current_da = (
                residuals.sel({config.SEX_DIM: sex}) 
                if config.SEX_DIM in residuals.dims 
                else residuals
            )
            for country in countries:
                country_da = (
                    current_da.sel({config.COUNTRY_DIM: country}) 
                    if config.COUNTRY_DIM in country_da.dims and country is not None
                    else current_da
                )

                title_suffix = f" - {sex}" if sex else ""
                if country:
                    title_suffix += f", {country}"
                curr_ax_settings = {"title": f"In-sample Residuals{title_suffix}"} | ax_settings

                self._plot_heatmap_from_matrix(
                    rates=country_da,
                    ax=axs[i],
                    ax_settings=curr_ax_settings, 
                    colorbar_settings=colorbar_settings, 
                    imshow_settings=imshow_settings
                )
                i += 1
        return axs

    def plot_year_snapshot(
            self,
            year: int,
            axs: Sequence[Axes] | None = None,
            ax_settings: dict[str, Any] = {},
            scatter_settings: dict[str, Any] = {},
            line_settings: dict[str, Any] = {}
        ) -> list[Axes]:
        countries = self._model._bound_dims.get(config.COUNTRY_DIM, [None])
        sexes = self._model._bound_dims.get(config.SEX_DIM, [None])

        axs = self._validate_and_normalize_axs(
            axs, len(sexes), ax_settings
        )

        actual_all = np.log(self._model.M.sel({config.YEAR_DIM: year}))
        predicted_all = np.log(self._model.predict_in_sample().sel({config.YEAR_DIM: year}))

        for i, sex in enumerate(sexes):
            current_actual = (
                actual_all.sel({config.SEX_DIM: sex}) 
                if config.SEX_DIM in actual_all.dims 
                else actual_all
            )
            current_pred = (
                predicted_all.sel({config.SEX_DIM: sex}) 
                if config.SEX_DIM in predicted_all.dims 
                else predicted_all
            )

            for c_idx, country in enumerate(countries):
                country_actual = (
                    current_actual.sel({config.COUNTRY_DIM: country}) 
                    if config.COUNTRY_DIM in current_actual.dims and country is not None
                    else current_actual
                )
                country_pred = (
                    current_pred.sel({config.COUNTRY_DIM: country}) 
                    if config.COUNTRY_DIM in current_pred.dims and country is not None
                    else current_pred
                )

                x_dim = config.AGE_DIM
                x_axis = country_actual.coords[x_dim].values

                label_suffix = f" ({country})" if country else ""
                
                has_countries = countries[0] is not None
                curr_scatter_settings = {
                    k: (v[c_idx] if isinstance(v, list) and has_countries else (v[i] if isinstance(v, list) else v)) 
                    for k, v in scatter_settings.items()
                }
                scatter_defaults = {
                    "color": "black",
                    "label": f"Observed {year}{label_suffix}",
                }
                axs[i].scatter(x_axis, country_actual.values, **(scatter_defaults | curr_scatter_settings))

                curr_line_settings = {
                    k: (v[c_idx] if isinstance(v, list) and has_countries else (v[i] if isinstance(v, list) else v)) 
                    for k, v in line_settings.items()
                }
                line_defaults = {"label": f"Prediction{label_suffix}"}
                axs[i].plot(x_axis, country_pred.values, **(line_defaults | curr_line_settings))

            ax_defaults = {
                "xlabel": f"{config.AGE_DIM} ${config.PLOTTING_LABELS.get(config.AGE_DIM, '')}$",
                "ylabel": f"Log-mortalities ({sex})" if sex else "Log-mortalities"
            }
            axs[i].set(**(ax_defaults | self._get_setting(ax_settings, i)))
            if countries[0] is not None:
                axs[i].legend()
        return axs

    def plot_fitted_vs_actual(
            self, 
            axs: Sequence[Axes] | None = None,
            ax_settings: dict[str, Any] = {},
            axline_settings: dict[str, Any] = {},
            scatter_settings: dict[str, Any] = {}
        ) -> list[Axes]:
        countries = self._model._bound_dims.get(config.COUNTRY_DIM, [None])
        sexes = self._model._bound_dims.get(config.SEX_DIM, [None])

        axs = self._validate_and_normalize_axs(
            axs, len(sexes), ax_settings
        )

        predicted_all = self._model.predict_in_sample()

        for i, sex in enumerate(sexes):
            current_actual = (
                self._model.M.sel({config.SEX_DIM: sex}) 
                if config.SEX_DIM in self._model.M.dims 
                else self._model.M
            )
            current_pred = (
                predicted_all.sel({config.SEX_DIM: sex}) 
                if config.SEX_DIM in predicted_all.dims 
                else predicted_all
            )

            min_value = float('inf')
            max_value = float('-inf')

            for c_idx, country in enumerate(countries):
                country_actual = (
                    current_actual.sel({config.COUNTRY_DIM: country}) 
                    if config.COUNTRY_DIM in current_actual.dims and country is not None
                    else current_actual
                )
                country_pred = (
                    current_pred.sel({config.COUNTRY_DIM: country}) 
                    if config.COUNTRY_DIM in current_pred.dims and country is not None
                    else current_pred
                )

                act_vals = country_actual.values.ravel()
                pred_vals = country_pred.values.ravel()

                min_value = min(min_value, act_vals.min(), pred_vals.min())
                max_value = max(max_value, act_vals.max(), pred_vals.max())

                has_countries = countries[0] is not None
                curr_scatter_settings = {
                    k: (v[c_idx] if isinstance(v, list) and has_countries else (v[i] if isinstance(v, list) else v)) 
                    for k, v in scatter_settings.items()
                }
                scatter_defaults = {
                    "label": country if country else None,
                    "alpha": 0.25
                }
                axs[i].scatter(act_vals, pred_vals, **(scatter_defaults | curr_scatter_settings))

            edge_space = (max_value - min_value) * 0.1 if max_value > min_value else 0.1
            max_value += edge_space
            min_value -= edge_space

            axline_defaults = {
                "color": "black",
                "linestyle": "dashed"
            }
            axs[i].axline(
                [min_value, min_value], 
                [max_value, max_value],
                **(axline_defaults | axline_settings)
            )

            ax_defaults = {
                "xlabel": "Actual mortalities",
                "ylabel": f"Predicted mortalities ({sex})" if sex else "Predicted mortalities",
                "xlim": [min_value, max_value],
                "ylim": [min_value, max_value]
            }
            axs[i].set(**(ax_defaults | self._get_setting(ax_settings, i)))
            if countries[0] is not None:
                axs[i].legend()
        return axs

    # ====================================================== #
    # ==================== GLM PLOTTERS ==================== #
    # ====================================================== #

    def plot_residual_heatmap(
            self,
            residual_type: Literal["deviance", "pearson"],
            axs: Sequence[Axes] | None = None,
            ax_settings: dict[str, Any] = {},
            colorbar_settings: dict[str, Any] = {},
            imshow_settings: dict[str, Any] = {}
        ) -> list[Axes]:
        countries = self._model._bound_dims.get(config.COUNTRY_DIM, [None])
        sexes = self._model._bound_dims.get(config.SEX_DIM, [None])

        axs = self._validate_and_normalize_axs(
            axs, len(sexes) * len(countries), ax_settings
        )

        residual_function_call = residual_type + "_residuals"
        proper_residual_label = residual_type[0].upper() + residual_type[1:]

        residuals = getattr(self._model, residual_function_call)

        colorbar_settings = {"label": f"{proper_residual_label} residuals"} | colorbar_settings

        i = 0
        for sex in sexes:
            current_da = (
                residuals.sel({config.SEX_DIM: sex}) 
                if config.SEX_DIM in residuals.dims 
                else residuals
            )
            for country in countries:
                country_da = (
                    current_da.sel({config.COUNTRY_DIM: country}) 
                    if config.COUNTRY_DIM in country_da.dims and country is not None
                    else current_da
                )

                title_suffix = f" - {sex}" if sex else ""
                if country:
                    title_suffix += f", {country}"
                curr_ax_settings = {"title": f"{proper_residual_label} Residuals{title_suffix}"} | ax_settings

                self._plot_heatmap_from_matrix(
                    rates=country_da,
                    ax=axs[i],
                    ax_settings=curr_ax_settings,
                    colorbar_settings=colorbar_settings,
                    imshow_settings=imshow_settings
                )
                i += 1
        return axs

    def plot_residual_histogram(
            self, 
            residual_type: Literal["deviance", "pearson"],
            axs: Sequence[Axes] | None = None,
            ax_settings: dict[str, Any] = {},
            hist_settings: dict[str, Any] = {}
        ) -> list[Axes]:
        countries = self._model._bound_dims.get(config.COUNTRY_DIM, [None])
        sexes = self._model._bound_dims.get(config.SEX_DIM, [None])

        axs = self._validate_and_normalize_axs(
            axs, len(sexes), ax_settings
        )

        residual_function_call = residual_type + "_residuals"
        proper_residual_label = residual_type[0].upper() + residual_type[1:]

        residuals = getattr(self._model, residual_function_call)

        for i, sex in enumerate(sexes):
            current_da = (
                residuals.sel({config.SEX_DIM: sex}) 
                if config.SEX_DIM in residuals.dims 
                else residuals
            )

            for c_idx, country in enumerate(countries):
                country_da = (
                    current_da.sel({config.COUNTRY_DIM: country}) 
                    if config.COUNTRY_DIM in country_da.dims and country is not None
                    else current_da
                )
                vals = country_da.values.ravel()

                has_countries = countries[0] is not None
                curr_hist_settings = {
                    k: (v[c_idx] if isinstance(v, list) and has_countries else (v[i] if isinstance(v, list) else v)) 
                    for k, v in hist_settings.items()
                }
                hist_defaults = {
                    "bins": 30, 
                    "alpha": 0.5,
                    "edgecolor": "black",
                    "label": country if country else None
                }
                axs[i].hist(vals, **(hist_defaults | curr_hist_settings))

            ax_defaults = {
                "xlabel": f"{proper_residual_label} residual value",
                "ylabel": f"Frequency ({sex})" if sex else "Frequency"
            }
            axs[i].set(**(ax_defaults | self._get_setting(ax_settings, i)))
            if countries[0] is not None:
                axs[i].legend()
        return axs

    def plot_residual_scatter(
            self,
            residual_type: Literal["deviance", "pearson"],
            axs: Sequence[Axes] | None = None,
            ax_settings: dict[str, Any] = {},
            scatter_settings: dict[str, Any] = {}
        ) -> list[Axes]:
        countries = self._model._bound_dims.get(config.COUNTRY_DIM, [None])
        sexes = self._model._bound_dims.get(config.SEX_DIM, [None])

        axs = self._validate_and_normalize_axs(
            axs, len(sexes), ax_settings
        )

        residual_function_call = residual_type + "_residuals"
        proper_residual_label = residual_type[0].upper() + residual_type[1:]

        ln_mortalities_x = np.log(self._model.predict_in_sample())
        residuals_y = getattr(self._model, residual_function_call)

        for i, sex in enumerate(sexes):
            current_x = (
                ln_mortalities_x.sel({config.SEX_DIM: sex}) 
                if config.SEX_DIM in ln_mortalities_x.dims 
                else ln_mortalities_x
            )
            current_y = (
                residuals_y.sel({config.SEX_DIM: sex}) 
                if config.SEX_DIM in residuals_y.dims 
                else residuals_y
            )

            for c_idx, country in enumerate(countries):
                country_x = (
                    current_x.sel({config.COUNTRY_DIM: country}) 
                    if config.COUNTRY_DIM in current_x.dims and country is not None
                    else current_x
                )
                country_y = (
                    current_y.sel({config.COUNTRY_DIM: country}) 
                    if config.COUNTRY_DIM in current_y.dims and country is not None
                    else current_y
                )

                has_countries = countries[0] is not None
                curr_scatter_settings = {
                    k: (v[c_idx] if isinstance(v, list) and has_countries else (v[i] if isinstance(v, list) else v)) 
                    for k, v in scatter_settings.items()
                }
                scatter_defaults = {
                    "label": country if country else None,
                    "alpha": 0.25
                }
                axs[i].scatter(
                    country_x.values.ravel(), 
                    country_y.values.ravel(),
                    **(scatter_defaults | curr_scatter_settings)
                )

            ax_defaults = {
                "xlabel": "$\\ln(m_{x,t})$",
                "ylabel": f"{proper_residual_label} residuals ({sex})" if sex else f"{proper_residual_label} residuals"
            }
            axs[i].set(**(ax_defaults | self._get_setting(ax_settings, i)))
            if countries[0] is not None:
                axs[i].legend()
        return axs
    
    def plot_convergence(
        self,
        axs: Sequence[Axes] | None = None,
        ax_settings: dict[str, Any] = {},
        line_settings: dict[str, Any] = {}
    ) -> list[Axes]:
        assert_glm(self._model)
        countries = self._model._bound_dims.get(config.COUNTRY_DIM, [None])
        sexes = self._model._bound_dims.get(config.SEX_DIM, [None])

        axs = self._validate_and_normalize_axs(axs, len(sexes), ax_settings)

        for i, sex in enumerate(sexes):
            for c_idx, country in enumerate(countries):
                seq = []
                for da in self._model.ll_history_:
                    current = (
                        da.sel({config.SEX_DIM: sex})
                        if config.SEX_DIM in da.dims and sex is not None 
                        else da
                    )
                    current = (
                        current.sel({config.COUNTRY_DIM: country}) 
                        if config.COUNTRY_DIM in current.dims and country is not None 
                        else current
                    )
                    seq.append(
                        float(current.values.item() if current.ndim == 0 
                        else current.values)
                    )

                line_defaults = {"marker": "o"}
                if country is not None:
                    line_defaults["label"] = str(country)

                has_countries = countries[0] is not None
                curr_line_settings = {
                    k: (v[c_idx] if isinstance(v, list) and has_countries else (v[i] if isinstance(v, list) else v)) 
                    for k, v in line_settings.items()
                }
                axs[i].plot(np.arange(len(seq)), seq, **(line_defaults | curr_line_settings))

            ax_defaults = {
                "xlabel": "Iterations",
                "ylabel": f"Log-likelihood $\\ell(\\cdot)$ value ({sex})" if sex else "Log-likelihood $\\ell(\\cdot)$ value"
            }
            axs[i].set(**(ax_defaults | self._get_setting(ax_settings, i)))
            
            if countries[0] is not None:
                axs[i].legend()
                
        return axs