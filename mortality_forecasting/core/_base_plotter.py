from collections.abc import Sequence
from abc import ABC
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import xarray as xr

from mortality_forecasting import config


class Plotter(ABC):
    def _validate_and_normalize_axs(
            self,
            axes_user_input: list[Axes] | Axes | None,
            axs_needed: int = 1
        ) -> list[Axes] | Axes:
        if axes_user_input is not None and isinstance(axes_user_input, Sequence):
            if len(axes_user_input) != axs_needed:
                raise ValueError(
                    f"This plotting function needs {axs_needed} " \
                    f"axs, while you provided only {len(axes_user_input)}."
                )
            axs = axes_user_input
        elif isinstance(axes_user_input, Axes):
            axs = [axes_user_input]
        else:
            axs = [plt.subplots()[1] for _ in range(axs_needed)]
        return axs[0] if axs_needed==1 else axs

    def _validate_settings_length(self, axs_needed: int = 1, *settings) -> None:
        if axs_needed == 1:
            return

        for setting in settings:
            if not setting:
                continue
            for key, value in setting.items():
                error_msg = f"Please provide {axs_needed} different values for '{key}'"
                if isinstance(value, Sequence) and not isinstance(value, str):
                    if len(value) != axs_needed:
                        raise ValueError(error_msg)
                else:
                    raise ValueError(error_msg)

    def _plot_heatmap_from_matrix(
            self,
            rates: xr.DataArray,
            ax: Axes,
            ax_settings: dict[str, Any],
            colorbar_settings: dict[str, Any],
            imshow_settings: dict[str, Any]
        ) -> Axes:
        x_axis_ages = rates.coords[config.AGE_DIM].values
        y_axis_years = rates.coords[config.YEAR_DIM].values

        imshow_defaults = {"origin": "lower", "cmap": "magma"}
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
            "label": f"Value distribution"
        }
        ax.figure.colorbar(
            im, 
            ax=ax, 
            **(colorbar_defaults | colorbar_settings)
        )
        return ax  