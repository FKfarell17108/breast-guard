import os
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


CANONICAL_FEATURES: List[str] = [
    "age",
    "menarche",
    "menopause",
    "agefirst",
    "children",
    "breastfeeding",
    "nrelbc",
    "imc",
    "weight",
    "exercise",
    "alcohol",
    "tobacco",
    "allergies",
]


COLNAME_MAP: Dict[str, str] = {
    "age": "age",
    "edad": "age",
    "umur": "age",
    "years": "age",
    "age(years)": "age",
    "menarche": "menarche",
    "ageatmenarche": "menarche",
    "age at menarche": "menarche",
    "agemenarche": "menarche",
    "menopause": "menopause",
    "menopausal": "menopause",
    "is_menopause": "menopause",
    "agefirst": "agefirst",
    "age at first birth": "agefirst",
    "agefirstchild": "agefirst",
    "age at first childbirth": "agefirst",
    "children": "children",
    "numchildren": "children",
    "parity": "children",
    "number of children": "children",
    "breastfeeding": "breastfeeding",
    "breast feeding": "breastfeeding",
    "lactation": "breastfeeding",
    "lactating": "breastfeeding",
    "breast": "breast",
    "side": "breast",
    "nrelbc": "nrelbc",
    "nrelativesbc": "nrelbc",
    "family_history_bc": "nrelbc",
    "relatives_bc": "nrelbc",
    "bmi": "imc",
    "imc": "imc",
    "body mass index": "imc",
    "weight": "weight",
    "peso": "weight",
    "exercise": "exercise",
    "physical_activity": "exercise",
    "physical activity": "exercise",
    "alcohol": "alcohol",
    "alcohol_intake": "alcohol",
    "tobacco": "tobacco",
    "smoke": "tobacco",
    "smoking": "tobacco",
    "allergies": "allergies",
    "allergy": "allergies",
    "cancer": "cancer",
    "target": "cancer",
    "diagnosis": "cancer",
}


BIN_TRUE = {"yes", "y", "si", "sí", "true", "1", 1, True, "t"}
BIN_FALSE = {"no", "n", "false", "0", 0, False, "f"}


def _clean_column_names(columns: List[str]) -> List[str]:
    cleaned: List[str] = []
    for name in columns:
        key = str(name).strip().lower() if name is not None else name
        if key is None:
            cleaned.append(key)
            continue
        key = key.replace("-", " ").replace("_", " ")
        key = " ".join(key.split())
        mapped = COLNAME_MAP.get(key, key.replace(" ", ""))
        cleaned.append(mapped)
    return cleaned


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalize_binary(value: Any) -> float:
    if pd.isna(value):
        return np.nan 
    s = str(value).strip().lower()
    if s in BIN_TRUE:
        return 1.0
    if s in BIN_FALSE:
        return 0.0
    if s in {"right", "left"}:
        return np.nan 
    return np.nan 


def normalize_binary_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return df
    df[col] = df[col].map(_normalize_binary)
    return df


def fix_breastfeeding_and_side(df: pd.DataFrame) -> pd.DataFrame:
    side_col = None
    if "breast" in df.columns:
        side_col = "breast"
    elif "breast_side" in df.columns:
        side_col = "breast_side"
    if side_col is not None:
        side = df[side_col].astype(str).str.strip().str.lower()
        side = side.replace({
            "right": "right",
            "r": "right",
            "left": "left",
            "l": "left",
            "bilateral": "bilateral",
            "both": "bilateral",
        })
        df["breast_side"] = side
    if "breastfeeding" in df.columns:
        df = normalize_binary_column(df, "breastfeeding")
    return df


def load_all_datasets(dataset_dir: str) -> List[pd.DataFrame]:
    dataframes: List[pd.DataFrame] = []
    for fname in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, fname)
        if os.path.isdir(path):
            continue
        try:
            if fname.lower().endswith(".csv"):
                df = pd.read_csv(path)
            elif fname.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(path)
            elif fname.lower().endswith(".zip"):
                df = pd.read_csv(path, compression="zip")
            else:
                continue
            if df is not None and not df.empty:
                dataframes.append(df)
        except Exception:
            continue
    return dataframes


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = _clean_column_names(list(df.columns))
    for col in df.columns:
        if col in {"age", "menarche", "agefirst", "children", "imc", "weight", "nrelbc"}:
            df[col] = df[col].astype(str).str.extract(r"([-+]?[0-9]*\.?[0-9]+)", expand=False)
            df[col] = _to_numeric(df[col])
    for col in ["menopause", "breastfeeding", "exercise", "alcohol", "tobacco", "allergies"]:
        if col in df.columns:
            df = normalize_binary_column(df, col)
    df = fix_breastfeeding_and_side(df)
    if "cancer" in df.columns:
        df["cancer"] = df["cancer"].map(_normalize_binary)
        if df["cancer"].isna().any():
            df["cancer"] = df["cancer"].fillna(_to_numeric(df["cancer"]))
        df["cancer"] = df["cancer"].round().clip(0, 1)
    keep = [c for c in CANONICAL_FEATURES + ["cancer", "breast_side"] if c in df.columns]
    return df[keep]


def merge_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame(columns=CANONICAL_FEATURES + ["cancer", "breast_side"])
    std = [standardize_dataframe(df) for df in dfs]
    merged = pd.concat(std, axis=0, ignore_index=True)
    merged = merged.drop_duplicates()
    feature_cols = [c for c in CANONICAL_FEATURES if c in merged.columns]
    if feature_cols:
        merged = merged.dropna(how="all", subset=feature_cols)
    return merged


def impute_encode_scale(
    df: pd.DataFrame,
    features: List[str] = CANONICAL_FEATURES,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    artifacts: Dict[str, Any] = {
        "numeric_imputers": {},
        "categorical_imputers": {},
        "label_encoders": {},
        "scaler": None,
        "features": features,
        "categorical_columns": [],
        "numeric_columns": [],
    }
    for col in features:
        if col not in df.columns:
            df[col] = np.nan
    numeric_cols = ["age", "menarche", "agefirst", "children", "imc", "weight"]
    categorical_binary = ["menopause", "breastfeeding", "exercise", "alcohol", "tobacco", "allergies"]
    artifacts["numeric_columns"] = numeric_cols
    artifacts["categorical_columns"] = categorical_binary
    for col in numeric_cols:
        if df[col].notna().any():
            imp = SimpleImputer(strategy="median")
        else:
            imp = SimpleImputer(strategy="constant", fill_value=0.0)
        df[[col]] = imp.fit_transform(df[[col]])
        artifacts["numeric_imputers"][col] = imp
    for col in categorical_binary:
        if df[col].notna().any():
            imp = SimpleImputer(strategy="most_frequent")
        else:
            imp = SimpleImputer(strategy="constant", fill_value=0.0)
        df[[col]] = imp.fit_transform(df[[col]])
        df[col] = df[col].map(_normalize_binary)
        df[col] = df[col].fillna(0.0)
        artifacts["categorical_imputers"][col] = imp
    if "breast_side" in df.columns:
        le = LabelEncoder()
        df["breast_side"] = df["breast_side"].fillna("unknown").astype(str)
        df["breast_side_le"] = le.fit_transform(df["breast_side"])
        artifacts["label_encoders"]["breast_side"] = le
        if "breast_side_le" not in features:
            features = features + ["breast_side_le"]
            artifacts["features"] = features
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    artifacts["scaler"] = scaler
    return df, artifacts


def apply_artifacts_to_input(
    user_input: Dict[str, Any], artifacts: Dict[str, Any]
) -> pd.DataFrame:
    features: List[str] = artifacts["features"]
    row = {col: user_input.get(col, np.nan) for col in CANONICAL_FEATURES}
    df = pd.DataFrame([row])
    for col in artifacts.get("categorical_columns", []):
        if col in df.columns:
            df[col] = df[col].map(_normalize_binary)
    for col, imp in artifacts.get("numeric_imputers", {}).items():
        if col in df.columns:
            df[[col]] = imp.transform(df[[col]])
    for col, imp in artifacts.get("categorical_imputers", {}).items():
        if col in df.columns:
            df[[col]] = imp.transform(df[[col]])
            df[col] = df[col].map(_normalize_binary).fillna(0.0)
    if "breast_side" in df.columns and "breast_side" in artifacts.get("label_encoders", {}):
        le: LabelEncoder = artifacts["label_encoders"]["breast_side"]
        df["breast_side"] = df["breast_side"].fillna("unknown").astype(str)
        known = set(le.classes_)
        df["breast_side"] = df["breast_side"].apply(lambda x: x if x in known else "unknown")
        df["breast_side_le"] = le.transform(df["breast_side"])
    scaler: StandardScaler = artifacts["scaler"]
    numeric_cols = artifacts.get("numeric_columns", [])
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    final_df = df.reindex(columns=[c for c in features if c != "cancer"], fill_value=np.nan)
    return final_df

import os
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


CANONICAL_FEATURES: List[str] = [
    "age",
    "menarche",
    "menopause",
    "agefirst",
    "children",
    "breastfeeding",
    "nrelbc",
    "imc",
    "weight",
    "exercise",
    "alcohol",
    "tobacco",
    "allergies",
]


COLNAME_MAP: Dict[str, str] = {
    # Age
    "age": "age",
    "edad": "age",
    "umur": "age",
    "years": "age",
    "age(years)": "age",
    "menarche": "menarche",
    "ageatmenarche": "menarche",
    "age at menarche": "menarche",
    "agemenarche": "menarche",
    "menopause": "menopause",
    "menopausal": "menopause",
    "is_menopause": "menopause",
    "agefirst": "agefirst",
    "age at first birth": "agefirst",
    "agefirstchild": "agefirst",
    "age at first childbirth": "agefirst",
    "children": "children",
    "numchildren": "children",
    "parity": "children",
    "number of children": "children",
    "breastfeeding": "breastfeeding",
    "breast feeding": "breastfeeding",
    "lactation": "breastfeeding",
    "lactating": "breastfeeding",
    "breast": "breast", 
    "side": "breast",
    "nrelbc": "nrelbc",
    "nrelativesbc": "nrelbc",
    "family_history_bc": "nrelbc",
    "relatives_bc": "nrelbc",
    "bmi": "imc",
    "imc": "imc",
    "body mass index": "imc",
    "weight": "weight",
    "peso": "weight",
    "exercise": "exercise",
    "physical_activity": "exercise",
    "physical activity": "exercise",
    "alcohol": "alcohol",
    "alcohol_intake": "alcohol",
    "tobacco": "tobacco",
    "smoke": "tobacco",
    "smoking": "tobacco",
    "allergies": "allergies",
    "allergy": "allergies",
    "cancer": "cancer",
    "target": "cancer",
    "diagnosis": "cancer",
}


BIN_TRUE = {
    "yes",
    "y",
    "si",
    "sí",
    "true",
    "1",
    1,
    True,
    "t",
}
BIN_FALSE = {"no", "n", "false", "0", 0, False, "f"}


def _clean_column_names(columns: List[str]) -> List[str]:
    cleaned: List[str] = []
    for name in columns:
        if name is None:
            cleaned.append(name)
            continue
        key = str(name).strip().lower()
        key = key.replace("-", " ").replace("_", " ")
        key = " ".join(key.split())
        mapped = COLNAME_MAP.get(key, key.replace(" ", ""))
        cleaned.append(mapped)
    return cleaned


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalize_binary(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    # Common spelling noise
    s = s.replace("", "")
    if s in BIN_TRUE:
        return 1.0
    if s in BIN_FALSE:
        return 0.0
    # language variants
    if s in {"right", "left"}:
        return np.nan
    return np.nan


def normalize_binary_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return df
    df[col] = df[col].map(_normalize_binary)
    return df


def fix_breastfeeding_and_side(df: pd.DataFrame) -> pd.DataFrame:
    side_col = None
    if "breast" in df.columns:
        side_col = "breast"
    elif "breast_side" in df.columns:
        side_col = "breast_side"

    if side_col is not None:
        side = df[side_col].astype(str).str.strip().str.lower()
        side = side.replace({
            "right": "right",
            "r": "right",
            "left": "left",
            "l": "left",
            "bilateral": "bilateral",
            "both": "bilateral",
        })
        df["breast_side"] = side

    if "breastfeeding" in df.columns:
        df = normalize_binary_column(df, "breastfeeding")

    return df


def load_all_datasets(dataset_dir: str) -> List[pd.DataFrame]:
    dataframes: List[pd.DataFrame] = []
    for fname in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, fname)
        if os.path.isdir(path):
            continue
        try:
            if fname.lower().endswith(".csv"):
                df = pd.read_csv(path)
            elif fname.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(path)
            elif fname.lower().endswith(".zip"):
                # Best-effort: try to find a csv inside zip via pandas
                df = pd.read_csv(path, compression="zip")
            else:
                continue
            if df is not None and not df.empty:
                dataframes.append(df)
        except Exception:
            continue
    return dataframes


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = _clean_column_names(list(df.columns))

    for col in df.columns:
        if col in {"age", "menarche", "agefirst", "children", "imc", "weight", "nrelbc"}:
            df[col] = (
                df[col]
                .astype(str)
                .str.extract(r"([-+]?[0-9]*\.?[0-9]+)", expand=False)
            )
            df[col] = _to_numeric(df[col])

    for col in ["menopause", "breastfeeding", "exercise", "alcohol", "tobacco", "allergies"]:
        if col in df.columns:
            df = normalize_binary_column(df, col)

    df = fix_breastfeeding_and_side(df)

    if "cancer" in df.columns:
        df["cancer"] = df["cancer"].map(_normalize_binary)
        if df["cancer"].isna().any():
            df["cancer"] = df["cancer"].fillna(_to_numeric(df["cancer"]))
        df["cancer"] = df["cancer"].round().clip(0, 1)

    keep = [c for c in CANONICAL_FEATURES + ["cancer", "breast_side"] if c in df.columns]
    return df[keep]


def merge_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame(columns=CANONICAL_FEATURES + ["cancer", "breast_side"])
    std = [standardize_dataframe(df) for df in dfs]
    merged = pd.concat(std, axis=0, ignore_index=True)
    merged = merged.drop_duplicates()
    feature_cols = [c for c in CANONICAL_FEATURES if c in merged.columns]
    if feature_cols:
        merged = merged.dropna(how="all", subset=feature_cols)
    return merged


def impute_encode_scale(
    df: pd.DataFrame,
    features: List[str] = CANONICAL_FEATURES,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()

    artifacts: Dict[str, Any] = {
        "numeric_imputers": {},
        "categorical_imputers": {},
        "label_encoders": {},
        "scaler": None,
        "features": features,
        "categorical_columns": [],
        "numeric_columns": [],
    }

    for col in features:
        if col not in df.columns:
            df[col] = np.nan

    numeric_cols = ["age", "menarche", "agefirst", "children", "imc", "weight"]
    categorical_binary = ["menopause", "breastfeeding", "exercise", "alcohol", "tobacco", "allergies"]

    artifacts["numeric_columns"] = numeric_cols
    artifacts["categorical_columns"] = categorical_binary

    for col in numeric_cols:
        if df[col].notna().any():
            imp = SimpleImputer(strategy="median")
        else:
            imp = SimpleImputer(strategy="constant", fill_value=0.0)
        df[[col]] = imp.fit_transform(df[[col]])
        artifacts["numeric_imputers"][col] = imp

    for col in categorical_binary:
        if df[col].notna().any():
            imp = SimpleImputer(strategy="most_frequent")
        else:
            imp = SimpleImputer(strategy="constant", fill_value=0.0)
        df[[col]] = imp.fit_transform(df[[col]])
        df[col] = df[col].map(_normalize_binary)
        df[col] = df[col].fillna(0.0)
        artifacts["categorical_imputers"][col] = imp

    if "breast_side" in df.columns:
        le = LabelEncoder()
        df["breast_side"] = df["breast_side"].fillna("unknown").astype(str)
        df["breast_side_le"] = le.fit_transform(df["breast_side"])
        artifacts["label_encoders"]["breast_side"] = le
        if "breast_side_le" not in features:
            features = features + ["breast_side_le"]
            artifacts["features"] = features

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    artifacts["scaler"] = scaler

    return df, artifacts


def apply_artifacts_to_input(
    user_input: Dict[str, Any], artifacts: Dict[str, Any]
) -> pd.DataFrame:
    features: List[str] = artifacts["features"]
    row = {col: user_input.get(col, np.nan) for col in CANONICAL_FEATURES}
    df = pd.DataFrame([row])

    for col in artifacts.get("categorical_columns", []):
        if col in df.columns:
            df[col] = df[col].map(_normalize_binary)

    for col, imp in artifacts.get("numeric_imputers", {}).items():
        if col in df.columns:
            df[[col]] = imp.transform(df[[col]])

    for col, imp in artifacts.get("categorical_imputers", {}).items():
        if col in df.columns:
            df[[col]] = imp.transform(df[[col]])
            df[col] = df[col].map(_normalize_binary).fillna(0.0)

    if "breast_side" in df.columns and "breast_side" in artifacts.get("label_encoders", {}):
        le: LabelEncoder = artifacts["label_encoders"]["breast_side"]
        df["breast_side"] = df["breast_side"].fillna("unknown").astype(str)
        known = set(le.classes_)
        df["breast_side"] = df["breast_side"].apply(lambda x: x if x in known else "unknown")
        df["breast_side_le"] = le.transform(df["breast_side"])

    scaler: StandardScaler = artifacts["scaler"]
    numeric_cols = artifacts.get("numeric_columns", [])
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    final_df = df.reindex(columns=[c for c in features if c != "cancer"], fill_value=np.nan)
    return final_df


