import os
import json
from typing import Dict, Any, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import (
    CANONICAL_FEATURES,
    load_all_datasets,
    merge_dataframes,
    impute_encode_scale,
)


try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


MODELS_DIR = os.path.join("models")
REPORTS_DIR = os.path.join("reports")
DATASET_DIR = os.path.join("dataset")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def build_final_df() -> pd.DataFrame:
    dfs = load_all_datasets(DATASET_DIR)
    merged = merge_dataframes(dfs)
    return merged


def train_and_evaluate(final_df: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
    # Ensure target exists and is binary
    if "cancer" not in final_df.columns:
        raise ValueError("Target 'cancer' not found in merged data")

    # Drop rows with missing/invalid target before casting
    final_df = final_df.copy()
    valid_targets = {0, 1, 0.0, 1.0}
    final_df = final_df[final_df["cancer"].isin(valid_targets)].dropna(subset=["cancer"])  
    if final_df.empty:
        raise ValueError("No valid target values after cleaning; ensure 'cancer' contains 0/1")

    # Split features/target
    feature_cols = [c for c in CANONICAL_FEATURES if c in final_df.columns]
    # Include engineered breast_side_le if exists
    if "breast_side_le" in final_df.columns and "breast_side_le" not in feature_cols:
        feature_cols = feature_cols + ["breast_side_le"]

    X = final_df[feature_cols].copy()
    y = final_df["cancer"].astype(int).clip(0, 1)

    # Train-test split with stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit imputers/encoders/scaler on train only
    train_df = pd.concat([X_train, y_train.rename("cancer")], axis=1)
    X_train_proc, artifacts = impute_encode_scale(train_df)
    features = artifacts["features"]

    # Align columns to features for X_train
    X_train_final = X_train_proc[[c for c in features if c != "cancer"]]

    # Apply same artifacts to X_test
    from preprocessing import apply_artifacts_to_input

    def row_to_dict(row: pd.Series) -> Dict[str, Any]:
        return row.to_dict()

    test_rows = [row_to_dict(r) for _, r in X_test.iterrows()]
    X_test_list = [apply_artifacts_to_input(r, artifacts) for r in test_rows]
    X_test_final = pd.concat(X_test_list, axis=0).reset_index(drop=True)

    # Initialize and train XGBoost (with imbalance handling + tuning)
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed. Please add it to requirements and install.")
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    scale_pos_weight = max(1.0, neg / max(1.0, pos))

    base_model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        min_child_weight=1.0,
        gamma=0.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )
    param_dist = {
        "n_estimators": [300, 400, 600, 800],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.07, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_lambda": [1.0, 2.0, 4.0],
        "min_child_weight": [1.0, 2.0, 5.0],
        "gamma": [0.0, 0.1, 0.2],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring="average_precision",
        cv=cv,
        verbose=0,
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_train_final.values, y_train.values)
    model = search.best_estimator_

    # Predictions
    y_prob = model.predict_proba(X_test_final.values)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Reports
    report = classification_report(y_test, y_pred, zero_division=0)
    with open(os.path.join(REPORTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # Confusion matrix heatmap
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Cancer", "Cancer"], yticklabels=["No Cancer", "Cancer"]) 
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"))
    plt.close()

    # ROC curve + optimal threshold via Youden's J
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    optimal_threshold = float(thresholds[best_idx])
    plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "roc_curve.png"))
    plt.close()

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "optimal_threshold": optimal_threshold,
    }

    return (model, artifacts, features), metrics


def save_artifacts(bundle: Tuple[Any, Dict[str, Any], List[str]]) -> None:
    model, artifacts, features = bundle
    joblib.dump(model, os.path.join(MODELS_DIR, "xgb_model.joblib"))
    joblib.dump(artifacts, os.path.join(MODELS_DIR, "preprocess_artifacts.joblib"))
    import pickle
    with open(os.path.join(MODELS_DIR, "xgb_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODELS_DIR, "preprocess_artifacts.pkl"), "wb") as f:
        pickle.dump(artifacts, f)
    try:
        import h5py
        with h5py.File(os.path.join(MODELS_DIR, "xgb_model.h5"), "w") as h5f:
            booster_dump = model.get_booster().get_dump(with_stats=True)
            text = "\n".join(booster_dump)
            import numpy as _np
            h5f.create_dataset("booster_dump", data=_np.string_(text))
    except Exception:
        pass
    with open(os.path.join(MODELS_DIR, "features.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)


def main():
    final_df = build_final_df()
    bundle, metrics = train_and_evaluate(final_df)
    save_artifacts(bundle)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

import os
import json
from typing import Dict, Any, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import (
    CANONICAL_FEATURES,
    load_all_datasets,
    merge_dataframes,
    impute_encode_scale,
)


try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


MODELS_DIR = os.path.join("models")
REPORTS_DIR = os.path.join("reports")
DATASET_DIR = os.path.join("dataset")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def build_final_df() -> pd.DataFrame:
    dfs = load_all_datasets(DATASET_DIR)
    merged = merge_dataframes(dfs)
    return merged


def train_and_evaluate(final_df: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
    # Ensure target exists and is binary
    if "cancer" not in final_df.columns:
        raise ValueError("Target 'cancer' not found in merged data")

    # Drop rows with missing/invalid target before casting
    final_df = final_df.copy()
    valid_targets = {0, 1, 0.0, 1.0}
    final_df = final_df[final_df["cancer"].isin(valid_targets)].dropna(subset=["cancer"])  
    if final_df.empty:
        raise ValueError("No valid target values after cleaning; ensure 'cancer' contains 0/1")

    # Split features/target
    feature_cols = [c for c in CANONICAL_FEATURES if c in final_df.columns]
    # Include engineered breast_side_le if exists
    if "breast_side_le" in final_df.columns and "breast_side_le" not in feature_cols:
        feature_cols = feature_cols + ["breast_side_le"]

    X = final_df[feature_cols].copy()
    y = final_df["cancer"].astype(int).clip(0, 1)

    # Train-test split with stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit imputers/encoders/scaler on train only
    train_df = pd.concat([X_train, y_train.rename("cancer")], axis=1)
    X_train_proc, artifacts = impute_encode_scale(train_df)
    features = artifacts["features"]

    # Align columns to features for X_train
    X_train_final = X_train_proc[[c for c in features if c != "cancer"]]

    # Apply same artifacts to X_test
    from preprocessing import apply_artifacts_to_input

    def row_to_dict(row: pd.Series) -> Dict[str, Any]:
        return row.to_dict()

    test_rows = [row_to_dict(r) for _, r in X_test.iterrows()]
    X_test_list = [apply_artifacts_to_input(r, artifacts) for r in test_rows]
    X_test_final = pd.concat(X_test_list, axis=0).reset_index(drop=True)

    # Initialize and train XGBoost (with imbalance handling + tuning)
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed. Please add it to requirements and install.")
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    scale_pos_weight = max(1.0, neg / max(1.0, pos))

    base_model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        min_child_weight=1.0,
        gamma=0.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )
    param_dist = {
        "n_estimators": [300, 400, 600, 800],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.07, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_lambda": [1.0, 2.0, 4.0],
        "min_child_weight": [1.0, 2.0, 5.0],
        "gamma": [0.0, 0.1, 0.2],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring="average_precision",
        cv=cv,
        verbose=0,
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_train_final.values, y_train.values)
    model = search.best_estimator_

    # Predictions
    y_prob = model.predict_proba(X_test_final.values)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Reports
    report = classification_report(y_test, y_pred, zero_division=0)
    with open(os.path.join(REPORTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # Confusion matrix heatmap
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Cancer", "Cancer"], yticklabels=["No Cancer", "Cancer"]) 
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"))
    plt.close()

    # ROC curve + optimal threshold via Youden's J
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    optimal_threshold = float(thresholds[best_idx])
    plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "roc_curve.png"))
    plt.close()

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "optimal_threshold": optimal_threshold,
    }

    return (model, artifacts, features), metrics


def save_artifacts(bundle: Tuple[Any, Dict[str, Any], List[str]]) -> None:
    model, artifacts, features = bundle
    # joblib
    joblib.dump(model, os.path.join(MODELS_DIR, "xgb_model.joblib"))
    joblib.dump(artifacts, os.path.join(MODELS_DIR, "preprocess_artifacts.joblib"))
    # pkl
    import pickle

    with open(os.path.join(MODELS_DIR, "xgb_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODELS_DIR, "preprocess_artifacts.pkl"), "wb") as f:
        pickle.dump(artifacts, f)

    # h5: save model dump as text into h5 for placeholder (XGBoost has native .json/.ubj,
    # but user requested .h5; we store booster dump as dataset)
    try:
        import h5py

        with h5py.File(os.path.join(MODELS_DIR, "xgb_model.h5"), "w") as h5f:
            booster_dump = model.get_booster().get_dump(with_stats=True)
            text = "\n".join(booster_dump)
            h5f.create_dataset("booster_dump", data=np.string_(text))
    except Exception:
        pass

    # Save features list
    with open(os.path.join(MODELS_DIR, "features.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)
    # Persist threshold if available in metrics_cache
    try:
        thr = 0.5
        if isinstance(metrics_cache, dict) and "optimal_threshold" in metrics_cache:
            thr = float(metrics_cache["optimal_threshold"])
        with open(os.path.join(MODELS_DIR, "risk_threshold.json"), "w", encoding="utf-8") as f:
            json.dump({"threshold": thr}, f)
    except Exception:
        pass


def main():
    final_df = build_final_df()
    bundle, metrics = train_and_evaluate(final_df)
    global metrics_cache
    metrics_cache = metrics
    save_artifacts(bundle)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()


