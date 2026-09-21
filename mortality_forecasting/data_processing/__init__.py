import json
from pathlib import Path

from mortality_forecasting.data_processing._dataset import MortalityDataset

_json_path = Path(__file__).parent / "supported_countries_fallback.json"
supported_countries_fallback = json.loads(_json_path.read_text())

__all__ = ["MortalityDataset", "supported_countries_fallback"]