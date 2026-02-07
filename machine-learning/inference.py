import json
import os
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

from preprocessing import apply_artifacts_to_input


MODELS_DIR = os.path.join("models")


class BreastCancerRiskModel:
    def __init__(self, models_dir: str = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self.model = joblib.load(os.path.join(models_dir, "xgb_model.joblib"))
        self.artifacts = joblib.load(os.path.join(models_dir, "preprocess_artifacts.joblib"))
        features_path = os.path.join(models_dir, "features.json")
        if os.path.exists(features_path):
            with open(features_path, "r", encoding="utf-8") as f:
                self.features = json.load(f)
        else:
            self.features = self.artifacts.get("features", [])

    def predict_risk(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        X = apply_artifacts_to_input(user_input, self.artifacts)
        probs = self.model.predict_proba(X.values)[:, 1]
        pred = (probs >= 0.5).astype(int)
        return {
            "pred": int(pred[0]),
            "proba": float(probs[0]),
        }


def _demo():
    user_input = {
        "age": 45,
        "menarche": 13,
        "menopause": 1,
        "agefirst": 22,
        "children": 2,
        "breastfeeding": 1,
        "nrelbc": 0,
        "imc": 23.5,
        "weight": 60,
        "exercise": 1,
        "alcohol": 0,
        "tobacco": 0,
        "allergies": 0,
    }
    model = BreastCancerRiskModel()
    print(model.predict_risk(user_input))


if __name__ == "__main__":
    _demo()


