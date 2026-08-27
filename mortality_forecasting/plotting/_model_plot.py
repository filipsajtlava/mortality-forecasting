from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from mortality_forecasting.core._base_plotter import Plotter
from mortality_forecasting import config


class ModelPlotter(Plotter):
    # TODO: this method needs a comprehensive docstring to explain the plot config
    def plot_parameters(
            self, 
            ax: Sequence[Axes] | None = None,
            plot_config: dict[str, Sequence[Any] | Any] = None
        ) -> list[Axes]:
        self.model._check_if_fitted()
        axs, plot_config = self._validate_inputs(
            ax, 
            plot_config, 
            axs_needed=len(self.model.parameters_)
        )

        for i, parameter in enumerate(self.model.parameters_):
            parameter_da = self.model.parameters_[parameter]
            x_dim = parameter_da.dims[0]
            x_axis = parameter_da.coords[x_dim].values

            individual_ax_config = {}
            if plot_config is not None:
                individual_ax_config = {
                    key: values[i] for key, values in plot_config.items()
                }

            if "label" not in individual_ax_config:
                individual_ax_config["label"] = self.model.value_column
                    
            axs[i].plot(x_axis, parameter_da, **individual_ax_config)
            axs[i].set(
                xlabel=f"{x_dim} {config.PLOTTING_LABELS[x_dim]}",
                ylabel=f"Parameter {parameter}"
            )
            axs[i].legend()
        return axs

    def plot_residual_heatmap(
            self,
            ax: Axes | None = None,
            plot_config: dict[str, Sequence[Any] | Any] = None
        ) -> Axes:
        self.model._check_if_fitted()
        ax, plot_config = self._validate_inputs(
            ax,
            plot_config, 
            additional_grids_required=("M",)
        )
        
        actual = self.model.mortality_data.M[self.model.value_column]
        predicted = self.model.predict_in_sample()
        residuals = (actual - predicted) / np.sqrt(predicted)

        x_axis_ages = residuals.coords[config.AGE_DIM]
        y_axis_years = residuals.coords[config.YEAR_DIM].values

        ax.imshow(
            residuals.T, 
            origin="lower",
            extent=[
                x_axis_ages.min(), x_axis_ages.max(), 
                y_axis_years.min(), y_axis_years.max()
            ], 
            **plot_config
        )
        ax.set(
            xlabel=f"{config.AGE_DIM} {config.PLOTTING_LABELS[config.AGE_DIM]}",
            ylabel=f"{config.YEAR_DIM} {config.PLOTTING_LABELS[config.YEAR_DIM]}"
        )
        im = ax.images[0]
        ax.figure.colorbar(
            im, 
            ax=ax, 
            label=f"Relative {self.model.value_column} mortality - deviance residuals"
        )
        return ax

    def plot_year_snapshot(
            self,
            year: int,
            ax: Axes | None = None,
            plot_config: dict[str, Sequence[Any] | Any] = None
        ):
        self.model._check_if_fitted()
        ax, plot_config = self._validate_inputs(
            ax,
            plot_config, 
            additional_grids_required=("M",)
        )

        actual = np.log(
            self.model.mortality_data.M[self.model.value_column]
            .sel({config.YEAR_DIM: year})
        )
        predicted = np.log(
            self.model.predict_in_sample()
            .sel({config.YEAR_DIM: year})
        )
        x_axis_ages = actual.coords[config.AGE_DIM].values

        if "label" not in plot_config:
            plot_config = plot_config.copy()
            plot_config["label"] = "Prediction"

        ax.scatter(
            x_axis_ages,
            actual,
            label=f"observed {year} values",
            color="black"
        )
        ax.plot(
            x_axis_ages,
            predicted,
            **plot_config
        )
        ax.set(
            xlabel=f"{config.AGE_DIM} {config.PLOTTING_LABELS[config.AGE_DIM]}",
            ylabel=f"Log-mortalities"
        )
        ax.legend()
        return ax

    def plot_fitted_vs_actual(
            self, 
            ax: Axes | None = None,
            plot_config: dict[str, Sequence[Any] | Any] = None,
            central_line: bool = True
        ) -> Axes:
        self.model._check_if_fitted()
        ax, plot_config = self._validate_inputs(
            ax, 
            plot_config, 
            additional_grids_required=("M",)
        )

        actual = self.model.mortality_data.M[self.model.value_column]
        predicted = self.model.predict_in_sample()

        max_value = max(actual.max(), predicted.max())
        min_value = min(actual.min(), predicted.min())
        edge_space = (max_value - min_value) * 0.1
        max_value += edge_space
        min_value -= edge_space

        ax.scatter(actual, predicted, **plot_config)
        if central_line:
            ax.axline(
                [min_value, min_value], 
                [max_value, max_value], 
                color="black", 
                linestyle="dashed"
            )
        ax.set(
            xlabel="Actual mortalities",
            ylabel="Predicted mortalities",
            xlim=[min_value, max_value], 
            ylim=[min_value, max_value]
        )
        return ax

    # TODO: think about moving these into the parent class, if possible
    def _validate_inputs(
            self, 
            axes_user_input: Sequence[Axes] | Axes | None, 
            plot_config: dict[str, Sequence[Any] | Any],
            additional_grids_required: tuple[str] | None = None,
            axs_needed: int = 1
        ) -> tuple[list[Axes] | Axes, dict[str, Sequence[Any] | Any]]:
        ax = self._validate_and_normalize_axs(axes_user_input, axs_needed)
        plot_config = self._validate_and_extrapolate_config(plot_config, axs_needed)

        if additional_grids_required is not None:
            self.model._validate_dataset(
                self.model.mortality_data,
                required_grids=additional_grids_required
            )

        if axs_needed == 1:
            ax = ax[0]
        return ax, plot_config

    def _validate_and_extrapolate_config(
            self, 
            plot_config: dict[str, Sequence[Any] | Any] | None,
            axs_needed: int
        ) -> dict[str, Any]:
        if not plot_config:
            return {}

        new_config = {}
        for key, settings in plot_config.items():
            is_seq = isinstance(settings, Sequence) and not isinstance(settings, str)

            if not is_seq:
                new_config[key] = settings if axs_needed == 1 else [settings] * axs_needed
            elif len(settings) != axs_needed:
                raise ValueError(
                    f"The configuration needs individual settings for "
                    f"each of the {axs_needed} axs, while you " 
                    f"provided {len(settings)} for '{key}'."
                )
            else:
                new_config[key] = settings[0] if axs_needed == 1 else settings

        return new_config

    def _validate_and_normalize_axs(
            self,
            axs: list[Axes] | Axes | None,
            axs_needed: int 
        ) -> list[Axes]:
        if axs is not None and isinstance(axs, Sequence):
            if len(axs) != axs_needed:
                raise ValueError(
                    f"This plotting function needs {axs_needed} " \
                    f"axs, while you provided only {len(axs)}."
                )
        elif isinstance(axs, Axes):
            axs = [axs]
        else:
            axs = []
            for _ in range(axs_needed):
                fig, ax = plt.subplots()
                axs.append(ax)
        return axs
