from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.base_model import Model


class Plotter(ABC):
    def __init__(self, model: Model) -> None:
        self.model = model
