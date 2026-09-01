from __future__ import annotations
from typing import Any, TYPE_CHECKING, Literal
from collections.abc import Sequence

from matplotlib.axes import Axes
from matplotlib.typing import ColorType
import xarray as xr
import numpy as np

from mortality_forecasting.core._base_plotter import Plotter
from mortality_forecasting import config
if TYPE_CHECKING:
    from mortality_forecasting.evaluation._evaluator import ForecastEvaluator


class EvaluatorPlotter(Plotter):
    def __init__(self, evaluator: ForecastEvaluator) -> None:
        self.evaluator = evaluator

    def plot_errors_by_age(
            self,
            training_data: xr.DataArray,
            ax: Axes | None = None,
            **kwargs
        ) -> Axes:
        ax = self._validate_and_normalize_axs(axes_user_input=ax)
        (
            ax_kwargs,
            under_kwargs, 
            over_kwargs,
            mase_kwargs,
            bar_kwargs
        ) = self._split_kwargs(
            1, 
            "ax", "underestimated", "overestimated", "mase", 
            **kwargs
        )

        mase_da = self.evaluator.mase(training_data)
        mser_da = self.evaluator.mser(training_data)

        under_defaults = {
            "color": "blue",
            "label": "Underestimated mortality"
        }
        over_defaults = {
            "color": "red",
            "label": "Overestimated mortality"
        }
        under_cfg = under_defaults | under_kwargs 
        over_cfg = over_defaults | over_kwargs

        mask = np.where(mser_da >= 0, under_cfg["color"], over_cfg["color"])
        ax.bar(np.arange(0, mase_da.size), mase_da.values, color=mask, **bar_kwargs)

        ax.plot(
            [], [], 
            **under_cfg,
            marker="s", linestyle=""
        )
        ax.plot(
            [], [], 
            **over_cfg,
            marker="s", linestyle=""
        )

        overall_mase = float(mase_da.mean())
        mase_defaults = {
            "color": "black",
            "label": f"MASE: {round(overall_mase, 3)}"
        }
        ax.plot(
            [], [],
            marker="s", linestyle="",
            **(mase_defaults | mase_kwargs)
        )

        ax.tick_params(axis="x", rotation=90)
        ax.set_xticks(np.arange(0, mase_da.size, 10))
        ax_defaults = {
            "xlabel": f"{config.AGE_DIM} {config.PLOTTING_LABELS[config.AGE_DIM]}",
            "ylabel": f"Value of MASE"
        }
        ax.set(**(ax_defaults | ax_kwargs))
        ax.legend()
        return ax

    def plot_forecasts(
            self,
            selected_ages: list[int],
            training_data: xr.DataArray,
            ax: Axes | None = None,
            colors: Sequence[ColorType] | None = None,
            **kwargs
        ) -> Axes:
        ax = self._validate_and_normalize_axs(axes_user_input=ax)
        (
            ax_kwargs, 
            training_kwargs, 
            testing_kwargs,
            fill_kwargs, 
            vline_kwargs,
            line_kwargs
        ) = self._split_kwargs(
            1, 
            "ax", "training", "testing", "fill", "vline",
            **kwargs
        )

        split_year = training_data[config.YEAR_DIM][-1]

        vline_defaults = {
            "color": "black",
            "linewidth": 2
        }
        ax.axvline(x=split_year, **(vline_defaults | vline_kwargs))
        # for the plot lines to intersect correctly, we have to extend the 
        # actual data, to contain the last year of the train set as well
        actual_back_extended = xr.concat(
            [training_data.isel({config.YEAR_DIM: -1}), self.evaluator.actual],
            dim=config.YEAR_DIM
        )
        forecast_back_extended = xr.concat(
            [training_data.isel({config.YEAR_DIM: -1}), self.evaluator.forecast],
            dim=config.YEAR_DIM
        )

        if colors is not None:
            ax.set_prop_cycle(color=colors)
        for i, age in enumerate(selected_ages): 
            
            if colors is not None:
                color = ax._get_lines.get_next_color()
            else:
                color = "black"
            training_defaults = {
                "color": color,
                "label": f"Age {age}",
                "linewidth": 2
            }
            ax.plot(
                training_data[config.YEAR_DIM],
                training_data.sel({config.AGE_DIM: age}),
                **(training_defaults | training_kwargs)
            )

            testing_defaults = {
                "color": color,
                "linewidth": 1,
                "alpha": 1
            }
            ax.plot(
                actual_back_extended[config.YEAR_DIM],
                actual_back_extended.sel({config.AGE_DIM: age}),
                **(testing_defaults | testing_kwargs)
            )

            line_defaults = {
                "color": color,
                "linestyle": "dashed"
            }
            ax.plot(
                forecast_back_extended[config.YEAR_DIM],
                forecast_back_extended.sel({
                    config.AGE_DIM: age,
                    config.BOUND_DIM: "point"
                }),
                **(line_defaults | line_kwargs)
            )

            fill_defaults = {
                "alpha": 0.25,
                "color": color
            }
            ax.fill_between(
                forecast_back_extended[config.YEAR_DIM],
                forecast_back_extended.sel({
                    config.AGE_DIM: age,
                    config.BOUND_DIM: "lower"
                }),
                forecast_back_extended.sel({
                    config.AGE_DIM: age,
                    config.BOUND_DIM: "upper"
                }),
                **(fill_defaults | fill_kwargs)
            )

        ax_defaults = {
            "xlabel": f"{config.YEAR_DIM} {config.PLOTTING_LABELS[config.YEAR_DIM]}",
            "ylabel": f"Mortality values"
        }
        ax.set(**(ax_defaults | ax_kwargs))
        ax.legend()
        return ax
        
    def _resolve_color(
            self, 
            user_color: ColorType | Sequence[ColorType] | None, 
            index: int, 
            total: int,
            default: ColorType
        ) -> ColorType:
        if user_color is None:
            return default
        
        if hasattr(user_color, "__len__") and not isinstance(user_color, str):
            if len(user_color) == total:
                return user_color[index]
                
        return user_color
            