from dataclasses import dataclass
from typing import Literal, Iterator

import xarray as xr

from mortality_forecasting.core._commons import ParameterContainer
from mortality_forecasting import config


@dataclass(frozen=True)
class ParamSpecification:
    name: str
    latex: str
    category: Literal["static", "period", "cohort"]
    naive_value: float
    dimension: str

class ModelParams:
    def __iter__(self) -> Iterator[ParamSpecification]:
        for attribute_name in dir(self):
            if not attribute_name.startswith("_"):
                param_spec = getattr(self, attribute_name)
                yield param_spec

    def _build_container(
            self,
            estimated_parameters: dict[ParamSpecification, xr.DataArray]
        ) -> ParameterContainer:
        def get_vars(category: str) -> dict[str, xr.DataArray]:
            return {
                spec.name: param_data.transpose(*config.DIM_ORDERING, missing_dims="ignore")
                for spec, param_data in estimated_parameters.items()
                if isinstance(spec, ParamSpecification) and spec.category == category
            }

        cohort_params = get_vars("cohort")            
        return ParameterContainer(
            static=xr.Dataset(get_vars("static")),
            period=xr.Dataset(get_vars("period")),
            cohort=xr.Dataset(cohort_params) if cohort_params else None
        )
        

class LeeCarterParams(ModelParams):
    AX = ParamSpecification("ax", "$\\alpha_x$", "static", 0.0, config.AGE_DIM)
    BX = ParamSpecification("bx", "$\\beta_x$", "static", 0.0, config.AGE_DIM)
    KT = ParamSpecification("kt", "$\\kappa_t$", "period", 1.0, config.YEAR_DIM)

class RenshawHabermanParams(ModelParams):
    AX = ParamSpecification("ax", "$\\alpha_x$", "static", 0.0, config.AGE_DIM)
    BX1 = ParamSpecification("bx1", "$\\beta_x^{(1)}$", "static", 0.0, config.AGE_DIM)
    BX2 = ParamSpecification("bx2", "$\\beta_x^{(2)}$", "static", 0.0, config.AGE_DIM)
    KT = ParamSpecification("kt", "$\\kappa_t$", "period", 1.0, config.YEAR_DIM)
    GC = ParamSpecification("gc", "$\\gamma_{t-x}$", "cohort", 1.0, config.COHORT_DIM)
