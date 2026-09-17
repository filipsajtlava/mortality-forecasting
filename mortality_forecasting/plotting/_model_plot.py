from __future__ import annotations
from typing import Any, Literal, TYPE_CHECKING
from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes

from mortality_forecasting import config
from mortality_forecasting.core._base_plotter import Plotter
from mortality_forecasting.core._commons import require_glm
if TYPE_CHECKING:
    from mortality_forecasting.core._base_model import Model


class ModelPlotter(Plotter):
    def __init__(self, model: Model) -> None:
        self.model = model

    # TODO: this method needs a comprehensive docstring to explain the plot config
    def plot_parameters(
            self, 
            axs: Sequence[Axes] | None = None,
            ax_settings: dict[str, Any] = {},
            line_settings: dict[str, Any] = {}
        ) -> list[Axes]:
        self.model._check_if_fitted()
        n_params = len(self.model.parameters_)
        axs = self._validate_and_normalize_axs(
            axes_user_input=axs,
            axs_needed=n_params,
        )
        self._validate_settings_length(n_params, ax_settings, line_settings)

        for i, parameter in enumerate(self.model.parameters_):
            parameter_da = self.model.parameters_[parameter]
            x_dim = parameter_da.dims[0]
            x_axis = parameter_da.coords[x_dim].values

            sub_line_settings = {k: v[i] for k, v in line_settings.items()}
            sub_line_defaults = {"label": self.model.value_column}
            axs[i].plot(x_axis, parameter_da, **(sub_line_defaults | sub_line_settings))

            sub_ax_settings = {k: v[i] for k, v in ax_settings.items()}
            sub_ax_defaults = {
                "xlabel": f"{x_dim} ${config.PLOTTING_LABELS[x_dim]}$",
                "ylabel": f"Parameter {parameter}",
            }
            axs[i].set(**(sub_ax_defaults | sub_ax_settings))
            axs[i].legend()
        return axs

    # This needs a proper docstring explaining the computation of the residual
    def plot_mortality_insample_heatmap(
            self,
            ax: Axes | None = None,
            ax_settings: dict[str, Any] = {},
            colorbar_settings: dict[str, Any] = {},
            imshow_settings: dict[str, Any] = {}
        ) -> Axes:
        self.model._check_if_fitted()
        ax = self._validate_and_normalize_axs(axes_user_input=ax)
        self._validate_settings_length(1, ax_settings, colorbar_settings, imshow_settings)
        
        actual = (
            self.model.mortality_data.D[self.model.value_column] /
            self.model.mortality_data.E[self.model.value_column]
        )
        predicted = self.model.predict_in_sample()
        residuals = (actual - predicted) / np.sqrt(predicted)

        colorbar_settings = (
            {"label": "Standardized mortality residuals"} | colorbar_settings
        )
        return self._plot_heatmap_from_matrix(
            rates=residuals,
            ax=ax,
            ax_settings=ax_settings, 
            colorbar_settings=colorbar_settings, 
            imshow_settings=imshow_settings
        )

    def plot_year_snapshot(
            self,
            year: int,
            ax: Axes | None = None,
            ax_settings: dict[str, Any] = {},
            scatter_settings: dict[str, Any] = {},
            line_settings: dict[str, Any] = {}
        ) -> Axes:
        self.model._check_if_fitted()
        ax = self._validate_and_normalize_axs(axes_user_input=ax)
        self._validate_settings_length(1, ax_settings, scatter_settings, line_settings)

        M = (
            self.model.mortality_data.D[self.model.value_column] /
            self.model.mortality_data.E[self.model.value_column]
        )
        actual = np.log(M.sel({config.YEAR_DIM: year}))
        predicted = np.log(
            self.model.predict_in_sample()
            .sel({config.YEAR_DIM: year})
        )
        x_axis_ages = actual.coords[config.AGE_DIM].values

        scatter_defaults = {
            "color": "black",
            "label": f"Observed {year} values",
        }
        ax.scatter(x_axis_ages, actual, **(scatter_defaults | scatter_settings))

        line_defaults = {"label": "Prediction"}
        ax.plot(x_axis_ages, predicted, **(line_defaults | line_settings))

        ax_defaults = {
            "xlabel": f"{config.AGE_DIM} ${config.PLOTTING_LABELS[config.AGE_DIM]}$",
            "ylabel": "Log-mortalities"
        }
        ax.set(**(ax_defaults | ax_settings))
        ax.legend()
        return ax

    def plot_fitted_vs_actual(
            self, 
            ax: Axes | None = None,
            ax_settings: dict[str, Any] = {},
            axline_settings: dict[str, Any] = {},
            scatter_settings: dict[str, Any] = {}
        ) -> Axes:
        self.model._check_if_fitted()
        ax = self._validate_and_normalize_axs(axes_user_input=ax)
        self._validate_settings_length(1, ax_settings, axline_settings, scatter_settings)

        actual = (
            self.model.mortality_data.D[self.model.value_column] /
            self.model.mortality_data.E[self.model.value_column]
        )
        predicted = self.model.predict_in_sample()

        max_value = max(actual.max(), predicted.max())
        min_value = min(actual.min(), predicted.min())
        edge_space = (max_value - min_value) * 0.1
        max_value += edge_space
        min_value -= edge_space

        ax.scatter(actual, predicted, **scatter_settings)

        axline_defaults = {
            "color": "black",
            "linestyle": "dashed"
        }
        ax.axline(
            [min_value, min_value], 
            [max_value, max_value],
            **(axline_defaults | axline_settings)
        )

        ax_defaults = {
            "xlabel": "Actual mortalities",
            "ylabel": "Predicted mortalities",
            "xlim": [min_value, max_value],
            "ylim": [min_value, max_value]
        }
        ax.set(**(ax_defaults | ax_settings))
        return ax

    # ====================================================== #
    # ==================== GLM PLOTTERS ==================== #
    # ====================================================== #

    @require_glm
    def plot_residual_heatmap(
            self,
            residual_type: Literal["deviance", "pearson"],
            ax: Axes | None = None,
            ax_settings: dict[str, Any] = {},
            colorbar_settings: dict[str, Any] = {},
            imshow_settings: dict[str, Any] = {}
        ) -> Axes:
        self.model._check_if_fitted()
        ax = self._validate_and_normalize_axs(axes_user_input=ax)
        self._validate_settings_length(1, ax_settings, colorbar_settings, imshow_settings)

        residual_function_call = residual_type + "_residuals"
        proper_residual_label = residual_type[0].upper() + residual_type[1:]

        residuals = getattr(self.model, residual_function_call)

        colorbar_settings = (
            {"label": f"{proper_residual_label} residuals"} | colorbar_settings
        )
        return self._plot_heatmap_from_matrix(
            rates=residuals,
            ax=ax,
            ax_settings=ax_settings,
            colorbar_settings=colorbar_settings,
            imshow_settings=imshow_settings
        )

    @require_glm
    def plot_residual_histogram(
            self, 
            residual_type: Literal["deviance", "pearson"],
            ax: Axes | None = None,
            ax_settings: dict[str, Any] = {},
            hist_settings: dict[str, Any] = {}
        ) -> Axes:
        self.model._check_if_fitted()
        ax = self._validate_and_normalize_axs(axes_user_input=ax)
        self._validate_settings_length(1, ax_settings, hist_settings)

        residual_function_call = residual_type + "_residuals"
        proper_residual_label = residual_type[0].upper() + residual_type[1:]

        residuals = getattr(self.model, residual_function_call)

        hist_defaults = {"bins": 30, "edgecolor": "black"}
        ax.hist(
            residuals.values.ravel(), 
            **(hist_defaults | hist_settings)
        )

        ax_defaults = {
            "xlabel": f"{proper_residual_label} residual value $r_{{x,t}}^{residual_type[0].upper()}$",
            "ylabel": "Frequency"
        }
        ax.set(**(ax_defaults | ax_settings))
        return ax

    @require_glm
    def plot_residual_scatter(
            self,
            residual_type: Literal["deviance", "pearson"],
            ax: Axes | None = None,
            ax_settings: dict[str, Any] = {},
            scatter_settings: dict[str, Any] = {}
        ) -> Axes:
        self.model._check_if_fitted()
        ax = self._validate_and_normalize_axs(axes_user_input=ax)
        self._validate_settings_length(1, ax_settings, scatter_settings)

        residual_function_call = residual_type + "_residuals"
        proper_residual_label = residual_type[0].upper() + residual_type[1:]

        ln_mortalities_x = np.log(self.model.predict_in_sample())
        residuals_y = getattr(self.model, residual_function_call)

        scatter_defaults = {}
        ax.scatter(
            ln_mortalities_x.values.ravel(), 
            residuals_y.values.ravel(),
            **(scatter_defaults | scatter_settings)
        )

        ax_defaults = {
            "xlabel": "$\\ln(m_{x,t})$",
            "ylabel": f"{proper_residual_label} residuals $r_{{x,t}}^{residual_type[0].upper()}$"
        }
        ax.set(**(ax_defaults | ax_settings))
        return ax

    @require_glm
    def plot_convergence(
            self,
            ax: Axes | None = None,
            ax_settings: dict[str, Any] = {},
            line_settings: dict[str, Any] = {}
        ) -> Axes:
        self.model._check_if_fitted()
        ax = self._validate_and_normalize_axs(axes_user_input=ax)
        self._validate_settings_length(1, ax_settings, line_settings)

        log_likelihood = self.model.log_likelihood_history_

        line_defaults = {"marker": "o"}
        ax.plot(
            np.arange(len(log_likelihood)),
            log_likelihood,
            **(line_defaults | line_settings)
        )

        ax_defaults = {
            "xlabel": "Iterations",
            "ylabel": "Log-likelihood $\\ell(\\cdot)$ value"
        }
        ax.set(**(ax_defaults | ax_settings))
        return ax