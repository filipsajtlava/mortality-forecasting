from __future__ import annotations
from typing import Any, TYPE_CHECKING
from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes

from mortality_forecasting import config
from mortality_forecasting.core._base_plotter import Plotter
if TYPE_CHECKING:
    from mortality_forecasting.core._base_model import Model


class ModelPlotter(Plotter):
    def __init__(self, model: Model) -> None:
        self.model = model

    # TODO: this method needs a comprehensive docstring to explain the plot config
    def plot_parameters(
            self, 
            axs: Sequence[Axes] | None = None,
            **kwargs
        ) -> list[Axes]:
        self.model._check_if_fitted()
        n_params = len(self.model.parameters_)
        axs = self._validate_and_normalize_axs(
            axes_user_input=axs,
            axs_needed=n_params,
        )
        ax_kwargs, line_kwargs = self._split_kwargs(n_params ,"ax", **kwargs)

        for i, parameter in enumerate(self.model.parameters_):
            parameter_da = self.model.parameters_[parameter]
            x_dim = parameter_da.dims[0]
            x_axis = parameter_da.coords[x_dim].values

            sub_line_kw = {k: v[i] for k, v in line_kwargs.items()}
            sub_line_defaults = {"label": self.model.value_column}
            axs[i].plot(x_axis, parameter_da, **(sub_line_defaults |sub_line_kw))

            sub_ax_kw = {k: v[i] for k, v in ax_kwargs.items()}
            sub_ax_defaults = {
                "xlabel": f"{x_dim} {config.PLOTTING_LABELS[x_dim]}",
                "ylabel": f"Parameter {parameter}",
            }
            axs[i].set(**(sub_ax_defaults | sub_ax_kw))
            axs[i].legend()
        return axs

    def plot_residual_heatmap(
            self,
            ax: Axes | None = None,
            **kwargs
        ) -> Axes:
        self.model._check_if_fitted()
        ax = self._validate_and_normalize_axs(axes_user_input=ax)
        ax_kwargs, colorbar_kwargs, imshow_kwargs = self._split_kwargs(
            1, "ax", "colorbar", **kwargs
        )
        
        actual = (
            self.model.mortality_data.D[self.model.value_column] /
            self.model.mortality_data.E[self.model.value_column]
        )
        predicted = self.model.predict_in_sample()
        residuals = (actual - predicted) / np.sqrt(predicted)

        x_axis_ages = residuals.coords[config.AGE_DIM]
        y_axis_years = residuals.coords[config.YEAR_DIM].values

        imshow_defaults = {"origin": "lower"}
        ax.imshow(
            residuals.T,
            extent=[
                x_axis_ages.min(), x_axis_ages.max(), 
                y_axis_years.min(), y_axis_years.max()
            ], 
            **(imshow_defaults | imshow_kwargs)
        )

        im = ax.images[0]
        ax_defaults = {
            "xlabel": f"{config.AGE_DIM} {config.PLOTTING_LABELS[config.AGE_DIM]}",
            "ylabel": f"{config.YEAR_DIM} {config.PLOTTING_LABELS[config.YEAR_DIM]}",
        }
        ax.set(**(ax_defaults | ax_kwargs))

        colorbar_defaults = {
            "label": f"Relative {self.model.value_column} mortality - deviance residuals"
        }
        ax.figure.colorbar(
            im, 
            ax=ax, 
            **(colorbar_defaults | colorbar_kwargs)
        )
        return ax

    def plot_year_snapshot(
            self,
            year: int,
            ax: Axes | None = None,
            **kwargs
        ) -> Axes:
        self.model._check_if_fitted()
        ax = self._validate_and_normalize_axs(axes_user_input=ax)
        ax_kwargs, scatter_kwargs, line_kwargs = self._split_kwargs(
            1, "ax", "scatter", **kwargs
        )

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
            "label": f"observed {year} values",
        }
        ax.scatter(x_axis_ages, actual, **(scatter_defaults | scatter_kwargs))

        line_defaults = {"label": "Prediction"}
        ax.plot(x_axis_ages, predicted, **(line_defaults | line_kwargs))

        ax_defaults = {
            "xlabel": f"{config.AGE_DIM} {config.PLOTTING_LABELS[config.AGE_DIM]}",
            "ylabel": "Log-mortalities"
        }
        ax.set(**(ax_defaults | ax_kwargs))
        ax.legend()
        return ax

    def plot_fitted_vs_actual(
            self, 
            ax: Axes | None = None,
            **kwargs
        ) -> Axes:
        self.model._check_if_fitted()
        ax = self._validate_and_normalize_axs(axes_user_input=ax)
        ax_kwargs, axline_kwargs, scatter_kwargs = self._split_kwargs(
            1, "ax", "axline", **kwargs
        )

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

        ax.scatter(actual, predicted, **scatter_kwargs)

        axline_defaults = {
            "color": "black",
            "linestyle": "dashed"
        }
        ax.axline(
            [min_value, min_value], 
            [max_value, max_value],
            **(axline_defaults | axline_kwargs)
        )

        ax_defaults = {
            "xlabel": "Actual mortalities",
            "ylabel": "Predicted mortalities",
            "xlim": [min_value, max_value],
            "ylim": [min_value, max_value]
        }
        ax.set(**(ax_defaults | ax_kwargs))
        return ax
