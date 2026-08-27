import json
from pathlib import Path

import numpy as np
import pytest

from models.poisson import PoissonModel
from data_downloading.datasets import MortalityDataset

BENCHMARK_DIR = Path(__file__).parent
INPUTS_DIR = BENCHMARK_DIR / "data"
OUTPUTS_DIR = BENCHMARK_DIR / "outputs"

@pytest.mark.parametrize(
    "dataset_prefix, initialization",
    [
        ("GBRTENW", "naive"),
        ("GBRTENW", "SVD"),
        ("SYNTHETIC", "naive"),
        ("SYNTHETIC", "SVD")
    ]
)
def test_against_stmomo_poisson(
        dataset_prefix: str,
        initialization: str
    ):
    dataset = MortalityDataset.load_from_files(
        D={"Total": INPUTS_DIR / f"{dataset_prefix}_Deaths.csv"},
        E={"Total": INPUTS_DIR / f"{dataset_prefix}_Exposures.csv"},
        M={"Total": INPUTS_DIR / f"{dataset_prefix}_Mortalities.csv"}
    )

    model = (
        PoissonModel(initialization=initialization)
        .fit(dataset, "Total")
    )

    param_file_dir = OUTPUTS_DIR / f"{dataset_prefix}_est_params_poisson.json"
    with param_file_dir.open("r", encoding="utf-8") as f:
        stmomo_parameters = json.load(f)

    for param in ["ax", "bx", "kt"]:
        np.testing.assert_allclose(
            model.parameters_[param], 
            stmomo_parameters[param], 
            rtol=1e-3, 
            atol=1e-4
        )