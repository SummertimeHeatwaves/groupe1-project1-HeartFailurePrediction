import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from typing import Optional, Tuple, Dict, List
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def load_data(filepath: str = "data/heart_failure.csv") -> pd.DataFrame:
    logger.info(f"Loading dataset from: {filepath}")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Dataset loaded successfully: {df.shape[0]} rows x {df.shape[1]} cols")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Memory BEFORE optimization: {start_mem:.4f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == np.float64:
            df[col] = df[col].astype(np.float32)
        elif col_type == np.int64:
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
                
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem
    logger.info(f"Memory AFTER optimization: {end_mem:.4f} MB")
    logger.info(f"Memory REDUCED by {reduction:.1f}%")
    return df

def check_missing_values(df: pd.DataFrame) -> pd.Series:
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing == 0:
        logger.info("No missing values found in the dataset.")
    else:
        logger.warning(f"Found {total_missing} missing values.")
    return missing

def handle_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None, method: str = "clip") -> pd.DataFrame:
    df_clean = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [c for c in columns if df[c].nunique() > 2]
        
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
        
    logger.info("Outlier handling complete.")
    return df_clean

def analyze_class_distribution(df: pd.DataFrame, target: str = "DEATH_EVENT") -> Dict:
    counts = df[target].value_counts()
    logger.info(f"Class Distribution for {target}:")
    logger.info(f" Survived (0): {counts.get(0, 0)}")
    logger.info(f" Deceased (1): {counts.get(1, 0)}")
    return {"counts": counts.to_dict()}

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    logger.info("Applying SMOTE...")
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE: {dict(y_res.value_counts())}")
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name=y_train.name)

def prepare_features(df: pd.DataFrame, target: str = "DEATH_EVENT", test_size: float = 0.2, apply_scaling: bool = True, apply_smote_flag: bool = True, random_state: int = 42) -> Dict:
    feature_cols = [c for c in df.columns if c not in [target, "time"]]
    X = df[feature_cols]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    scaler = None
    if apply_scaling:
        scaler = StandardScaler()
        X_train_processed = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=feature_cols)
        X_test_processed = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)
    else:
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()
        
    if apply_smote_flag:
        X_train_final, y_train_final = apply_smote(X_train_processed, y_train, random_state)
    else:
        X_train_final, y_train_final = X_train_processed, y_train
        
    return {
        "X_train": X_train_final, "X_test": X_test_processed,
        "y_train": y_train_final, "y_test": y_test,
        "scaler": scaler, "feature_names": feature_cols
    }

def run_preprocessing_pipeline(filepath: str = "data/heart_failure.csv") -> Dict:
    logger.info("STARTING PREPROCESSING PIPELINE")
    df = load_data(filepath)
    check_missing_values(df)
    df = optimize_memory(df)
    
    continuous_cols = ["creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium"]
    df = handle_outliers(df, columns=continuous_cols)
    analyze_class_distribution(df)
    
    data = prepare_features(df, apply_scaling=True, apply_smote_flag=True)
    logger.info("PREPROCESSING PIPELINE COMPLETE")
    return data

if __name__ == "__main__":
    data = run_preprocessing_pipeline()