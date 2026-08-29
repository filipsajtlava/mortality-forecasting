import json
from pathlib import Path

import numpy as np
import pytest

from mortality_forecasting.models import PoissonModel, LeeCarterModel
from mortality_forecasting.data_processing import MortalityDataset
from mortality_forecasting.forecasting import RandomWalkWithDrift
from mortality_forecasting import config
from mortality_forecasting.core._base_model import Model


BENCHMARK_DIR = Path(__file__).parent
INPUTS_DIR = BENCHMARK_DIR / "data"
OUTPUTS_DIR = BENCHMARK_DIR / "outputs"


def fit_model(dataset_prefix: str, model: Model) -> Model:
    dataset = MortalityDataset.load_from_files(
        D={"Total": INPUTS_DIR / f"{dataset_prefix}_Deaths.csv"},
        E={"Total": INPUTS_DIR / f"{dataset_prefix}_Exposures.csv"},
        M={"Total": INPUTS_DIR / f"{dataset_prefix}_Mortalities.csv"}
    )
    fitted_model = model.fit(dataset, "Total")
    return fitted_model

@pytest.mark.parametrize(
    "dataset_prefix",
    ["GBRTENW", "SYNTHETIC"]
)
def test_lc_estimation(dataset_prefix: str) -> None:
    param_file_dir = OUTPUTS_DIR / f"{dataset_prefix}_est_params_lc.json"
    with param_file_dir.open("r", encoding="utf-8") as f:
        stmomo_reference = json.load(f)
    model = fit_model(dataset_prefix, LeeCarterModel())

    for param in ["ax", "bx", "kt"]:
        np.testing.assert_allclose(
            model.parameters_[param],
            stmomo_reference["parameters"][param],
            rtol=1e-3,
            atol=1e-4
        )

@pytest.mark.parametrize(
    "dataset_prefix, initialization",
    [
        ("GBRTENW", "naive"),
        ("GBRTENW", "SVD"),
        ("SYNTHETIC", "naive"),
        ("SYNTHETIC", "SVD")
    ]
)
def test_poisson_estimation(
        dataset_prefix: str,
        initialization: str
    ) -> None:
    param_file_dir = OUTPUTS_DIR / f"{dataset_prefix}_est_params_poisson.json"
    with param_file_dir.open("r", encoding="utf-8") as f:
        stmomo_reference = json.load(f)
    model = fit_model(dataset_prefix, PoissonModel(initialization=initialization))

    for param in ["ax", "bx", "kt"]:
        np.testing.assert_allclose(
            model.parameters_[param], 
            stmomo_reference["parameters"][param], 
            rtol=1e-3, 
            atol=1e-4
        )

@pytest.mark.parametrize(
    "dataset_prefix, model",
    [
        ("GBRTENW", LeeCarterModel()),
        ("GBRTENW", PoissonModel()),
        ("SYNTHETIC", LeeCarterModel()),     
        ("SYNTHETIC", PoissonModel())
    ]
)
def test_rwd_forecast(dataset_prefix: str, model: Model) -> None:
    model_suffix = "poisson" if isinstance(model, PoissonModel) else "lc"
    param_file_dir = OUTPUTS_DIR / f"{dataset_prefix}_est_params_{model_suffix}.json"
    with param_file_dir.open("r", encoding="utf-8") as f:
        stmomo_parameters = json.load(f)
    model = fit_model(dataset_prefix, model)

    steps = len(stmomo_parameters["forecasts"]["kt"]["point"])
    forecasted_data = model.forecast(RandomWalkWithDrift(alpha=0.05), steps=steps)

    for bound in ["lower", "point", "upper"]:
        np.testing.assert_allclose(
            forecasted_data.parameters_["kt"].sel({config.BOUND_DIM: bound}),
            stmomo_parameters["forecasts"]["kt"][bound],
            rtol=1e-3,
            atol=1e-4
        )