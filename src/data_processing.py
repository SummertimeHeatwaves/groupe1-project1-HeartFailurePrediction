"""
data_processing.py
------------------
Core data processing pipeline for Heart Failure Prediction project.
Handles loading, cleaning, outlier treatment, feature engineering,
class imbalance, memory optimization, and train/test splitting.
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. LOADING
# ─────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV dataset and return a DataFrame."""
    df = pd.read_csv(filepath)
    print(f"[load_data] Loaded {df.shape[0]} rows × {df.shape[1]} columns.")
    return df


# ─────────────────────────────────────────────
# 2. BASIC VALIDATION
# ─────────────────────────────────────────────

def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate the dataframe and return a summary dict:
      - shape, dtypes, missing counts, class distribution.
    """
    report = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_counts": df.isnull().sum().to_dict(),
        "missing_total": int(df.isnull().sum().sum()),
        "class_distribution": df["DEATH_EVENT"].value_counts().to_dict(),
        "class_pct": (df["DEATH_EVENT"].value_counts(normalize=True) * 100).round(2).to_dict(),
    }
    print(f"[validate_data] Shape          : {report['shape']}")
    print(f"[validate_data] Missing values : {report['missing_total']}")
    print(f"[validate_data] Class dist.    : {report['class_distribution']}")
    print(f"[validate_data] Class %        : {report['class_pct']}")
    return report


# ─────────────────────────────────────────────
# 3. MISSING VALUE HANDLING
# ─────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values.
    Strategy:
      - Numerical  → median imputation (robust to outliers).
      - Binary/Cat → mode imputation.
    Note: This dataset has 0 missing values, but the function is
          written defensively to generalise to new data.
    """
    df = df.copy()
    binary_cols = [c for c in df.columns if df[c].nunique() <= 2]
    numerical_cols = [c for c in df.columns if c not in binary_cols + ["DEATH_EVENT"]]

    filled_num = 0
    filled_bin = 0

    for col in numerical_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            df[col] = df[col].fillna(df[col].median())
            filled_num += n_missing

    for col in binary_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            filled_bin += n_missing

    print(f"[handle_missing_values] Filled {filled_num} numerical, {filled_bin} binary missing values.")
    return df


# ─────────────────────────────────────────────
# 4. OUTLIER TREATMENT
# ─────────────────────────────────────────────

def detect_outliers_iqr(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Detect outliers using the IQR method and return a summary DataFrame.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [c for c in columns if c != "DEATH_EVENT"]

    records = []
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
        records.append({
            "feature": col, "Q1": Q1, "Q3": Q3, "IQR": IQR,
            "lower_bound": lower, "upper_bound": upper,
            "n_outliers": n_outliers,
            "outlier_pct": round(n_outliers / len(df) * 100, 2),
        })

    summary = pd.DataFrame(records)
    print(f"[detect_outliers_iqr] Outlier summary:\n{summary[['feature','n_outliers','outlier_pct']].to_string(index=False)}")
    return summary


def cap_outliers_iqr(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Cap (Winsorise) outliers to IQR bounds.
    Strategy: capping preserves all rows (important for small datasets like this one).
    Dropping would remove up to ~10% of data which is unacceptable here.
    """
    df = df.copy()
    if columns is None:
        columns = ["creatinine_phosphokinase", "platelets", "serum_creatinine", "serum_sodium", "ejection_fraction"]

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"[cap_outliers_iqr] {col}: capped {before} values to [{lower:.2f}, {upper:.2f}]")

    return df


# ─────────────────────────────────────────────
# 5. MEMORY OPTIMISATION
# ─────────────────────────────────────────────

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce DataFrame memory footprint by downcasting numeric columns:
      - float64 → float32
      - int64   → int32  (or int8 for binary columns)
    Returns the optimised DataFrame and prints memory delta.
    """
    df = df.copy()
    mem_before = df.memory_usage(deep=True).sum() / 1024  # KB

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)

    for col in df.select_dtypes(include=["int64"]).columns:
        if df[col].nunique() <= 2:
            df[col] = df[col].astype(np.int8)
        else:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            else:
                df[col] = df[col].astype(np.int32)

    mem_after = df.memory_usage(deep=True).sum() / 1024  # KB
    reduction = (1 - mem_after / mem_before) * 100
    print(f"[optimize_memory] Before: {mem_before:.2f} KB | After: {mem_after:.2f} KB | Reduction: {reduction:.1f}%")
    return df


# ─────────────────────────────────────────────
# 6. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create clinically meaningful derived features:
      - age_group          : categorical age bands
      - ef_severity        : ejection fraction severity class
      - creatinine_ratio   : serum_creatinine / median (normalised severity)
      - sodium_risk        : binary flag for hyponatremia (serum_sodium < 135)
      - cp_log             : log-transform of creatinine_phosphokinase (right-skewed)
      - platelet_log       : log-transform of platelets (right-skewed)
    """
    df = df.copy()

    # Age group
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 50, 60, 70, 100],
        labels=["<50", "50-60", "60-70", ">70"]
    ).astype(str)
    # Encode age_group as ordinal integer
    age_map = {"<50": 0, "50-60": 1, "60-70": 2, ">70": 3}
    df["age_group"] = df["age_group"].map(age_map)

    # Ejection fraction severity
    # Normal ≥55%, Mild 40-54%, Moderate 25-39%, Severe <25%
    def ef_category(ef):
        if ef >= 55:
            return 0  # Normal
        elif ef >= 40:
            return 1  # Mild
        elif ef >= 25:
            return 2  # Moderate
        else:
            return 3  # Severe
    df["ef_severity"] = df["ejection_fraction"].apply(ef_category)

    # Hyponatremia risk flag
    df["sodium_risk"] = (df["serum_sodium"] < 135).astype(int)

    # Log transform for highly skewed features
    df["cp_log"] = np.log1p(df["creatinine_phosphokinase"])
    df["platelet_log"] = np.log1p(df["platelets"])

    print(f"[feature_engineering] Added 5 features. New shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 7. CLASS IMBALANCE – OVERSAMPLING (Manual SMOTE)
# ─────────────────────────────────────────────

#def handle_class_imbalance(X: pd.DataFrame, y: pd.Series, method: str = "oversample", random_state: int = 42) -> tuple:
    """
    Handle class imbalance.

    Methods available:
      - 'oversample'    : random oversampling of minority class (no external lib needed)
      - 'class_weight'  : returns sample_weight array for use in model (recommended fallback)
      - 'none'          : no resampling (baseline)

    Note: SMOTE (preferred) requires imbalanced-learn. We implement random oversampling
    as a robust fallback that ships with zero extra dependencies. The train_model.py
    script can use class_weight='balanced' in sklearn models as an alternative.
    """
    if method == "none":
        print("[handle_class_imbalance] No resampling applied.")
        return X, y

    class_counts = y.value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    n_majority = class_counts[majority_class]
    n_minority = class_counts[minority_class]

    print(f"[handle_class_imbalance] Before: {dict(class_counts)}")

    if method == "oversample":
        # Random oversampling: duplicate minority samples with replacement
        rng = np.random.default_rng(random_state)
        minority_idx = y[y == minority_class].index
        n_to_add = n_majority - n_minority
        sampled_idx = rng.choice(minority_idx, size=n_to_add, replace=True)
        X_minority_up = X.loc[sampled_idx].copy()
        y_minority_up = y.loc[sampled_idx].copy()
        X_balanced = pd.concat([X, X_minority_up]).reset_index(drop=True)
        y_balanced = pd.concat([y, y_minority_up]).reset_index(drop=True)
        print(f"[handle_class_imbalance] After (oversample): {dict(y_balanced.value_counts())}")
        return X_balanced, y_balanced

    elif method == "class_weight":
        # Return sample weights for use in ML estimators
        n_total = len(y)
        n_classes = 2
        weights = {
            0: n_total / (n_classes * class_counts[0]),
            1: n_total / (n_classes * class_counts[1]),
        }
        sample_weights = y.map(weights).values
        print(f"[handle_class_imbalance] Class weights computed: {weights}")
        return X, y  # weights returned separately; caller should use sample_weight param

    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: oversample, class_weight, none")
def handle_class_imbalance(X_train, y_train, random_state=42):
    print(f"\n[handle_class_imbalance]")
    print(f"  Avant SMOTE : {dict(pd.Series(y_train).value_counts())}")

    smote = SMOTE(random_state=random_state)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)

    X_bal = pd.DataFrame(X_bal, columns=X_train.columns)
    y_bal = pd.Series(y_bal)

    print(f"  Après SMOTE : {dict(y_bal.value_counts())} ✅")
    return X_bal, y_bal


#def compute_class_weights(y: pd.Series) -> dict:
    """Compute balanced class weights dict for sklearn estimators."""
    class_counts = y.value_counts()
    n_total = len(y)
    n_classes = len(class_counts)
    weights = {cls: n_total / (n_classes * cnt) for cls, cnt in class_counts.items()}
    return weights


# ─────────────────────────────────────────────
# 8. FEATURE SELECTION
# ─────────────────────────────────────────────

def select_features(df: pd.DataFrame, drop_cols: list = None) -> tuple:
    """
    Separate features (X) and target (y).
    Drops highly correlated or leaky columns if specified.

    Key insight from EDA:
      - 'time' has the highest absolute correlation with DEATH_EVENT (r = -0.527).
        It represents follow-up period – in a real clinical tool we KEEP it as it
        reflects monitoring duration. Documented here for transparency.
      - No feature pair has |r| > 0.5 (multicollinearity not a concern).
    """
    if drop_cols is None:
        drop_cols = []

    feature_cols = [c for c in df.columns if c != "DEATH_EVENT" and c not in drop_cols]
    X = df[feature_cols]
    y = df["DEATH_EVENT"]
    print(f"[select_features] Features: {list(X.columns)}")
    print(f"[select_features] Target  : DEATH_EVENT | shape X={X.shape}, y={y.shape}")
    return X, y


# ─────────────────────────────────────────────
# 9. SCALING
# ─────────────────────────────────────────────

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Apply StandardScaler (fit on train, transform on both).
    Returns scaled arrays and the fitted scaler object.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"[scale_features] Scaling applied. Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────
# 10. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────

def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.2, random_state: int = 42) -> tuple:
    """Stratified train/test split preserving class ratios."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[split_data] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[split_data] Train class dist: {dict(y_train.value_counts())}")
    print(f"[split_data] Test  class dist: {dict(y_test.value_counts())}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 11. FULL PIPELINE
# ─────────────────────────────────────────────

def run_full_pipeline(filepath: str,
                      apply_feature_engineering: bool = True,
                      test_size: float = 0.2,
                      random_state: int = 42) -> dict:
    """
    End-to-end data pipeline:
      load → validate → missing values → outlier capping →
      memory optimisation → feature engineering →
      feature selection → train/test split →
      class imbalance handling (on train only) → scaling

    Returns a dict with all processed artefacts.
    """
    print("\n" + "="*60)
    print("  HEART FAILURE – FULL DATA PIPELINE")
    print("="*60)

    # 1. Load
    df = load_data(filepath)

    # 2. Validate
    report = validate_data(df)

    # 3. Missing values
    df = handle_missing_values(df)

    # 4. Outlier capping
    df = cap_outliers_iqr(df)

    # 5. Memory optimisation
    df = optimize_memory(df)

    # 6. Feature engineering
    if apply_feature_engineering:
        df = feature_engineering(df)

    # 7. Feature selection
    X, y = select_features(df)

    # 8. Train/test split (BEFORE resampling – test must stay pure)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)

    # 9. Handle imbalance on TRAIN ONLY
    X_train_bal, y_train_bal = handle_class_imbalance(X_train, y_train)

    # 10. Scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_bal, X_test)

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60 + "\n")

    return {
        "df_processed": df,
        "X": X, "y": y,
        "X_train": X_train_bal, "X_test": X_test,
        "y_train": y_train_bal, "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "validation_report": report,
    }


# ─────────────────────────────────────────────
# Quick smoke-test when run directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os
    DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/DATASET.csv")
    result = run_full_pipeline(DATA_PATH)
    print("Feature names:", result["feature_names"])
    print("X_train_scaled shape:", result["X_train_scaled"].shape)
    print("X_test_scaled shape :", result["X_test_scaled"].shape)

    
if __name__ == "__main__":
    import os
    DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/DATASET.csv")
    
    # Lancer le pipeline
    result = run_full_pipeline(DATA_PATH)
    
    # Sauvegarder la data traitée dans UN seul fichier
    df_processed = result["df_processed"]
    output_path = os.path.join(os.path.dirname(__file__), "../data/data_processed.csv")
    df_processed.to_csv(output_path, index=False)
    print(f"✅ Data traitée sauvegardée : {output_path}")
    print(f"   Shape : {df_processed.shape}")
    print(f"   Colonnes : {list(df_processed.columns)}")