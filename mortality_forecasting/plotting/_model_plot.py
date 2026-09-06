from __future__ import annotations
from typing import Any, TYPE_CHECKING
from collections.abc import Sequence
from functools import wraps

import numpy as np
from matplotlib.axes import Axes
import xarray as xr

from mortality_forecasting import config
from mortality_forecasting.core._base_plotter import Plotter
if TYPE_CHECKING:
    from mortality_forecasting.core._base_model import Model
from mortality_forecasting.core._base_glm import GLMCapable


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
                "xlabel": f"{x_dim} {config.PLOTTING_LABELS[x_dim]}",
                "ylabel": f"Parameter {parameter}",
            }
            axs[i].set(**(sub_ax_defaults | sub_ax_settings))
            axs[i].legend()
        return axs

    def plot_mortality_residual_heatmap(
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

        return self._plot_heatmap(
            residuals,
            ax,
            ax_settings, 
            colorbar_settings, 
            imshow_settings
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
            "xlabel": f"{config.AGE_DIM} {config.PLOTTING_LABELS[config.AGE_DIM]}",
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

    def _require_glm(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not isinstance(self.model, GLMCapable):
                raise TypeError(
                    f"'{func.__name__}' is only available for GLM structures."
                )
            return func(self, *args, **kwargs)
        return wrapper

    @_require_glm
    def plot_pearson_residual_heatmap(
            self,
            ax: Axes | None = None,
            ax_settings: dict[str, Any] = {},
            colorbar_settings: dict[str, Any] = {},
            imshow_settings: dict[str, Any] = {}
        ) -> Axes:
        self.model._check_if_fitted()
        ax = self._validate_and_normalize_axs(axes_user_input=ax)
        self._validate_settings_length(1, ax_settings, colorbar_settings, imshow_settings)

        pearson_residuals = self.model.get_pearson_residuals()
        return self._plot_heatmap(
            pearson_residuals, 
            ax,
            ax_settings, 
            colorbar_settings, 
            imshow_settings
        )

    @_require_glm
    def plot_deviance_residual_heatmap(
            self,
            ax: Axes | None = None,
            ax_settings: dict[str, Any] = {},
            colorbar_settings: dict[str, Any] = {},
            imshow_settings: dict[str, Any] = {}
        ) -> Axes:
        self.model._check_if_fitted()
        ax = self._validate_and_normalize_axs(axes_user_input=ax)
        self._validate_settings_length(1, ax_settings, colorbar_settings, imshow_settings)

        deviance_residuals = self.model.get_deviance_residuals()
        return self._plot_heatmap(
            deviance_residuals,
            ax,
            ax_settings, 
            colorbar_settings, 
            imshow_settings
        )

    def _plot_heatmap(
            self,
            rates: xr.DataArray,
            ax: Axes,
            ax_settings: dict[str, Any],
            colorbar_settings: dict[str, Any],
            imshow_settings: dict[str, Any]
        ) -> Axes:
        x_axis_ages = rates.coords[config.AGE_DIM]
        y_axis_years = rates.coords[config.YEAR_DIM].values

        imshow_defaults = {"origin": "lower"}
        ax.imshow(
            rates.T,
            extent=[
                x_axis_ages.min(), x_axis_ages.max(), 
                y_axis_years.min(), y_axis_years.max()
            ], 
            **(imshow_defaults | imshow_settings)
        )

        im = ax.images[0]
        ax_defaults = {
            "xlabel": f"{config.AGE_DIM} {config.PLOTTING_LABELS[config.AGE_DIM]}",
            "ylabel": f"{config.YEAR_DIM} {config.PLOTTING_LABELS[config.YEAR_DIM]}",
        }
        ax.set(**(ax_defaults | ax_settings))

        colorbar_defaults = {
            "label": f"Relative {self.model.value_column} mortality - deviance residuals"
        }
        ax.figure.colorbar(
            im, 
            ax=ax, 
            **(colorbar_defaults | colorbar_settings)
        )
        return ax    

    @_require_glm
    def plot_pearson_histogram(
            self, 
            ax: Axes | None = None,
            ax_settings: dict[str, Any] = {},
            hist_settings: dict[str, Any] = {}
        ) -> Axes:
        self.model._check_if_fitted()
        ax = self._validate_and_normalize_axs(axes_user_input=ax)
        self._validate_settings_length(1, ax_settings, hist_settings)

        ax_settings = {"xlabel": "Pearson residual value"} | ax_settings
        pearson_residuals = self.model.get_pearson_residuals()
        return self._plot_histogram(
            pearson_residuals,
            ax,
            ax_settings,
            hist_settings
        )

    @_require_glm
    def plot_deviance_histogram(
            self, 
            ax: Axes | None = None,
            ax_settings: dict[str, Any] = {},
            hist_settings: dict[str, Any] = {}
        ) -> Axes:
        self.model._check_if_fitted()
        ax = self._validate_and_normalize_axs(axes_user_input=ax)
        self._validate_settings_length(1, ax_settings, hist_settings)

        ax_settings = {"xlabel": "Deviance residual value"} | ax_settings
        deviance_residuals = self.model.get_deviance_residuals()
        return self._plot_histogram(
            deviance_residuals,
            ax,
            ax_settings,
            hist_settings
        )

    def _plot_histogram(
            self, 
            rates: xr.DataArray,
            ax: Axes,
            ax_settings: dict[str, Any],
            hist_settings: dict[str, Any]
        ) -> Axes:

        flat_rates = rates.values.ravel()
        hist_defaults = {"bins": 30}
        ax.hist(flat_rates, **(hist_settings | hist_defaults))

        ax_defaults = {
            "ylabel": "Frequency"
        }
        ax.set(**(ax_defaults | ax_settings))
        return ax