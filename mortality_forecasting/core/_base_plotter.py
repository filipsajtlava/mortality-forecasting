from typing import Any
from collections.abc import Sequence
from abc import ABC

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


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

    def _split_kwargs(
            self, 
            axs_needed: int = 1,
            *prefixes: str,
            **kwargs
        ) -> tuple[dict[str, Any], ...]:
        results = {prefix: {} for prefix in prefixes}
        unprefixed = {}

        for key, value in kwargs.items():
            matched_prefix = False
            for prefix in prefixes:
                prefix_tag = f"{prefix}_"
                if key.startswith(prefix_tag):
                    clean_key = key[len(prefix_tag):]
                    results[prefix][clean_key] = value
                    matched_prefix = True
                    break
            if not matched_prefix:
                unprefixed[key] = value

        return *(results[p] for p in prefixes), unprefixed