from abc import ABC

from core.base_model import Model


class Plotter(ABC):
    def __init__(self, model: Model) -> None:
        self.model = model

    def _check_if_fitted(self) -> None:
        parameters = getattr(self.model, "parameters_", None)
        if parameters is None:
            raise ValueError("You need to fit the model first.")