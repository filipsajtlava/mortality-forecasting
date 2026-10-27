from collections.abc import Sequence
from abc import ABC
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import xarray as xr

from mortality_forecasting import config


class Plotter(ABC):
    def _validate_and_normalize_axs(
            self,
            axes_user_input: list[Axes] | Axes | None,
            axs_needed: int,
            *settings: dict[str, Any]
        ) -> list[Axes]:
        self._validate_settings_length(axs_needed, *settings)

        if axes_user_input is not None and isinstance(axes_user_input, Sequence):
            if len(axes_user_input) != axs_needed:
                raise ValueError(
                    f"This plotting function needs {axs_needed} "
                    f"axs, while you provided only {len(axes_user_input)}."
                )
            axs = list(axes_user_input)
        elif isinstance(axes_user_input, Axes):
            if axs_needed != 1:
                raise ValueError(
                    f"This plotting function needs {axs_needed} axes, "
                    f"but a single Axes object was provided."
                )
            axs = [axes_user_input]
        else:
            axs = [plt.subplots()[1] for _ in range(axs_needed)]
        return axs

    def _validate_settings_length(self, axs_needed: int, *settings) -> None:
        for setting in settings:
            if not setting:
                continue
            for key, value in setting.items():
                is_sequence = isinstance(value, Sequence) and not isinstance(value, str)
                if is_sequence and len(value) != axs_needed:
                    raise ValueError(
                        f"Setting '{key}' has {len(value)} values, "
                        f"but exactly {axs_needed} are required (or a single broadcasting value)."
                    )

    def _get_setting(self, settings: dict[str, Any], index: int) -> dict[str, Any]:
        resolved = {}
        for k, v in settings.items():
            if isinstance(v, list):
                resolved[k] = v[index]
            else:
                resolved[k] = v
        return resolved

    def _plot_heatmap_from_matrix(
            self,
            rates: xr.DataArray,
            ax: Axes,
            ax_settings: dict[str, Any],
            colorbar_settings: dict[str, Any],
            imshow_settings: dict[str, Any],
            dimensions: Literal["age-year", "year-age"] = "age-year"
        ) -> Axes:
        imshow_defaults = {"origin": "lower", "cmap": "magma"}
        if dimensions == "age-year":
            x_axis = rates.coords[config.AGE_DIM].values
            y_axis = rates.coords[config.YEAR_DIM].values
        else:
            x_axis = rates.coords[config.YEAR_DIM].values
            y_axis = rates.coords[config.AGE_DIM].values
        ax.imshow(
            rates,
            extent=[
                x_axis.min(), x_axis.max(), 
                y_axis.min(), y_axis.max()
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
            "label": f"Value distribution"
        }
        ax.figure.colorbar(
            im, 
            ax=ax, 
            **(colorbar_defaults | colorbar_settings)
        )
        return ax  