from __future__ import annotations
from typing import Any, TYPE_CHECKING
from collections.abc import Sequence

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.typing import ColorType

from mortality_forecasting import config
from mortality_forecasting.core._base_plotter import Plotter
if TYPE_CHECKING:
    from mortality_forecasting.evaluation._evaluator import ForecastEvaluator


class EvaluatorPlotter(Plotter):
    def __init__(self, evaluator: ForecastEvaluator) -> None:
        self._evaluator = evaluator
        self._axs_needed = sum(len(value) for value in self._evaluator._bound_dims.values())

    # ====================================================== #
    # =================== HELPER METHODS =================== #
    # ====================================================== #

    def _get_bound_dims(self) -> tuple[list[Any], list[Any], bool]:
        countries = self._evaluator._bound_dims.get(config.COUNTRY_DIM, [None])
        sexes = self._evaluator._bound_dims.get(config.SEX_DIM, [None])
        return countries, sexes, countries[0] is not None

    def _slice_arrays(
            self,
            *arrays: xr.DataArray,
            sex: Any = None,
            country: Any = None
        ) -> tuple[xr.DataArray, ...]:
        sliced = []
        for da in arrays:
            res = da
            if sex is not None and config.SEX_DIM in res.dims:
                res = res.sel({config.SEX_DIM: sex})
            if country is not None and config.COUNTRY_DIM in res.dims:
                res = res.sel({config.COUNTRY_DIM: country})
            sliced.append(res)
        return tuple(sliced)

    def _resolve_settings(
            self,
            *settings_dicts: dict[str, Any],
            idx: int
        ) -> tuple[dict[str, Any], ...]:
        return tuple(
            {k: (v[idx] if isinstance(v, list) else v) for k, v in s.items()}
            for s in settings_dicts
        )

    # ====================================================== #
    # ================== PLOTTING METHODS ================== #
    # ====================================================== #

    def plot_mase_by_age(
            self,
            training_data: xr.DataArray | xr.Dataset,
            axs: Sequence[Axes] | None = None,
            ax_settings: dict[str, Any] = {},
            underestimated_settings: dict[str, Any] = {},
            overestimated_settings: dict[str, Any] = {},
            mase_settings: dict[str, Any] = {},
            bar_settings: dict[str, Any] = {}
        ) -> list[Axes]:
        countries, sexes, _ = self._get_bound_dims()
        axs = self._validate_and_normalize_axs(
            axs, len(sexes) * len(countries),
            ax_settings, underestimated_settings, overestimated_settings,
            mase_settings, bar_settings
        )

        training_data = self._evaluator._normalize_dataset_input(training_data)
        mase_all = self._evaluator.mase(training_data)
        mser_all = self._evaluator.mser(training_data)

        i = 0
        for sex in sexes:
            for country in countries:
                country_mase, country_mser = self._slice_arrays(
                    mase_all, mser_all, sex=sex, country=country
                )
                curr_under, curr_over, curr_mase, curr_bar = self._resolve_settings(
                    underestimated_settings, overestimated_settings,
                    mase_settings, bar_settings,
                    idx=i
                )

                under_cfg = {"color": "blue", "label": "Underestimated mortality"} | curr_under
                over_cfg = {"color": "red", "label": "Overestimated mortality"} | curr_over

                colors = [
                    under_cfg["color"] if val >= 0 else over_cfg["color"]
                    for val in country_mser.values
                ]
                axs[i].bar(
                    np.arange(0, country_mase.size),
                    country_mase.values,
                    color=colors,
                    **curr_bar
                )

                axs[i].plot([], [], **under_cfg, marker="s", linestyle="")
                axs[i].plot([], [], **over_cfg, marker="s", linestyle="")

                overall_mase = float(country_mase.mean())
                mase_defaults = {
                    "color": "black",
                    "label": f"Overall MASE: {round(overall_mase, 3)}"
                }
                axs[i].plot([], [], marker="s", linestyle="", **(mase_defaults | curr_mase))

                axs[i].tick_params(axis="x", rotation=90)
                axs[i].set_xticks(np.arange(0, country_mase.size, 10))

                ylabel = f"Value of MASE ({sex})" if sex else "Value of MASE"
                if country:
                    ylabel += f" - {country}"

                ax_defaults = {
                    "xlabel": f"{config.AGE_DIM} ${config.PLOTTING_LABELS.get(config.AGE_DIM, '')}$",
                    "ylabel": ylabel
                }
                axs[i].set(**(ax_defaults | self._get_setting(ax_settings, i)))
                axs[i].legend()
                i += 1
        return axs

    def plot_bootstrap_residual_scatter(
            self,
            mortality_rates: xr.DataArray,
            axs: Sequence[Axes] | None = None,
            ax_settings: dict[str, Any] = {},
            scatter_settings: dict[str, Any] = {}
        ) -> list[Axes]:
        countries, sexes, has_countries = self._get_bound_dims()
        axs = self._validate_and_normalize_axs(
            axs, len(sexes), ax_settings
        )

        residuals = self._evaluator.forecast_container.in_sample_residuals_
        if residuals is None:
            raise ValueError("Please provide a forecast container with bootstrap residuals")

        residuals_y = residuals.sel({config.BOUND_DIM: "point"})
        ln_mortalities_x = np.log(mortality_rates)

        for i, sex in enumerate(sexes):
            for c_idx, country in enumerate(countries):
                country_x, country_y = self._slice_arrays(
                    ln_mortalities_x, residuals_y, sex=sex, country=country
                )
                (curr_scatter,) = self._resolve_settings(
                    scatter_settings, idx=c_idx if has_countries else i
                )

                scatter_defaults = {"label": country if country else None, "alpha": 0.25}
                axs[i].scatter(
                    country_x.values.ravel(),
                    country_y.values.ravel(),
                    **(scatter_defaults | curr_scatter)
                )

            ax_defaults = {
                "xlabel": "$\\ln(m_{x,t})$",
                "ylabel": f"Values of residuals ({sex})" if sex else "Values of residuals"
            }
            axs[i].set(**(ax_defaults | self._get_setting(ax_settings, i)))
            if has_countries:
                axs[i].legend()
        return axs

    def plot_bootstrap_residual_heatmap(
            self,
            base_model_residuals: xr.DataArray,
            axs: Sequence[Axes] | None = None,
            ax_settings: dict[str, Any] = {},
            colorbar_settings: dict[str, Any] = {},
            imshow_settings: dict[str, Any] = {},
            contour_settings: dict[str, Any] = {}
        ) -> list[Axes]:
        countries, sexes, _ = self._get_bound_dims()
        axs = self._validate_and_normalize_axs(
            axs, len(sexes) * len(countries),
            ax_settings, colorbar_settings, imshow_settings, contour_settings
        )

        res_bounds = self._evaluator.forecast_container.in_sample_residuals_
        if res_bounds is None:
            raise ValueError("Please provide a forecast container with bootstrap residuals")

        diff_all = base_model_residuals - res_bounds.sel({config.BOUND_DIM: "point"})
        violations_all = (
            (base_model_residuals > res_bounds.sel({config.BOUND_DIM: "upper"}))
            | (base_model_residuals < res_bounds.sel({config.BOUND_DIM: "lower"}))
        )

        colorbar_settings = {"label": "Deviation magnitude from bootstrap median"} | colorbar_settings

        i = 0
        for sex in sexes:
            for country in countries:
                country_diff, country_viol = self._slice_arrays(
                    diff_all, violations_all, sex=sex, country=country
                )
                (curr_contour,) = self._resolve_settings(contour_settings, idx=i)

                title_suffix = f" - {sex}" if sex else ""
                if country:
                    title_suffix += f", {country}"
                curr_ax_settings = {"title": f"Bootstrap Residuals{title_suffix}"} | ax_settings

                max_val = float(np.nanmax(np.abs(country_diff.values)))
                curr_imshow = {"cmap": "magma", "vmin": -max_val, "vmax": max_val} | imshow_settings

                self._plot_heatmap_from_matrix(
                    rates=country_diff,
                    ax=axs[i],
                    ax_settings=curr_ax_settings,
                    colorbar_settings=colorbar_settings,
                    imshow_settings=curr_imshow,
                    dimensions="year-age"
                )

                contour_defaults = {"colors": "black", "linewidths": 1, "linestyles": "-"}
                axs[i].contour(
                    country_diff.coords[config.YEAR_DIM].values,
                    country_diff.coords[config.AGE_DIM].values,
                    country_viol.values,
                    levels=[0.5],
                    **(contour_defaults | curr_contour)
                )
                i += 1
        return axs

    def plot_forecasts(
            self,
            selected_ages: Sequence[int],
            training_data: xr.DataArray | xr.Dataset,
            axs: Sequence[Axes] | None = None,
            colors: Sequence[ColorType] | None = None,
            ax_settings: dict[str, Any] = {},
            training_settings: dict[str, Any] = {},
            testing_settings: dict[str, Any] = {},
            vline_settings: dict[str, Any] = {},
            interval_fill_settings: dict[str, Any] = {},
            line_settings: dict[str, Any] = {}
        ) -> list[Axes]:
        countries, sexes, _ = self._get_bound_dims()
        axs = self._validate_and_normalize_axs(
            axs, len(sexes) * len(countries),
            ax_settings, training_settings, testing_settings,
            vline_settings, interval_fill_settings, line_settings
        )
        training_data = self._evaluator._normalize_dataset_input(training_data)

        last_train_year_da = training_data.isel({config.YEAR_DIM: -1})
        split_year = last_train_year_da[config.YEAR_DIM]

        actual_back_extended = xr.concat(
            [last_train_year_da, self._evaluator.actual], dim=config.YEAR_DIM
        )
        forecast_back_extended = xr.concat(
            [last_train_year_da, self._evaluator.forecast_container.mortality_rates_],
            dim=config.YEAR_DIM
        )

        i = 0
        for sex in sexes:
            for country in countries:
                train_da, act_da, fc_da = self._slice_arrays(
                    training_data, actual_back_extended, forecast_back_extended,
                    sex=sex, country=country
                )
                curr_vline, curr_train, curr_test, curr_line, curr_fill = self._resolve_settings(
                    vline_settings, training_settings, testing_settings,
                    line_settings, interval_fill_settings,
                    idx=i
                )

                axs[i].axvline(x=split_year, **({"color": "black", "linewidth": 2} | curr_vline))

                if colors is not None:
                    axs[i].set_prop_cycle(color=colors)

                for age in selected_ages:
                    color = axs[i]._get_lines.get_next_color()
                    train_age, act_age, fc_age = (
                        da.sel({config.AGE_DIM: age}) for da in (train_da, act_da, fc_da)
                    )

                    axs[i].plot(
                        train_da[config.YEAR_DIM],
                        train_age,
                        **({"color": color, "label": f"Age ${age}$", "linewidth": 2} | curr_train)
                    )
                    axs[i].plot(
                        act_da[config.YEAR_DIM],
                        act_age,
                        **({"color": color, "linewidth": 1, "alpha": 1} | curr_test)
                    )
                    axs[i].plot(
                        fc_da[config.YEAR_DIM],
                        fc_age.sel({config.BOUND_DIM: "point"}),
                        **({"color": color, "linestyle": "dashed"} | curr_line)
                    )
                    axs[i].fill_between(
                        fc_da[config.YEAR_DIM],
                        fc_age.sel({config.BOUND_DIM: "lower"}),
                        fc_age.sel({config.BOUND_DIM: "upper"}),
                        **({"alpha": 0.25, "color": color} | curr_fill)
                    )

                ylabel = f"Mortality values ({sex})" if sex else "Mortality values"
                if country:
                    ylabel += f" - {country}"

                ax_defaults = {
                    "xlabel": f"{config.YEAR_DIM} ${config.PLOTTING_LABELS.get(config.YEAR_DIM, '')}$",
                    "ylabel": ylabel
                }
                axs[i].set(**(ax_defaults | self._get_setting(ax_settings, i)))
                axs[i].legend()
                i += 1
        return axs