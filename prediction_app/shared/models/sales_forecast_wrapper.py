from __future__ import annotations

from typing import Any

import numpy as np


class SalesForecastWrapper:
    """
    A tiny wrapper to keep model post-processing (e.g. log1p inverse) inside the pickled artifact.

    Why:
    - The API loads joblib models without separately reading metadata.json.
    - Some training setups use target transforms (log1p) to better handle wide value ranges
      (SKU vs Category totals). This wrapper ensures inference always returns values in the
      original business unit (e.g. revenue).
    """

    def __init__(self, model: Any, *, target_transform: str = "none"):
        self.model = model
        self.target_transform = (target_transform or "none").lower().strip()

        # Expose feature names for the API's feature alignment logic.
        if hasattr(model, "feature_names_in_"):
            self.feature_names_in_ = getattr(model, "feature_names_in_")

    def predict(self, X: Any):
        y = self.model.predict(X)
        if self.target_transform == "log1p":
            y = np.expm1(y)
        return y

