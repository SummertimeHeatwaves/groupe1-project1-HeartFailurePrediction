import unittest
import pandas as pd
import numpy as np
import sys
import os

# Ajouter le dossier src au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import (
    load_data,
    validate_data,
    handle_missing_values,
    cap_outliers_iqr,
    optimize_memory,
    feature_engineering,
    split_data,
    handle_class_imbalance,
    scale_features,
    run_full_pipeline
)

# Chemin vers la vraie data
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/DATASET.csv")


def make_sample_df():
    """Petite data synthétique pour les tests unitaires"""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "age":                     np.random.uniform(40, 90, n).astype(np.float64),
        "anaemia":                 np.random.randint(0, 2, n).astype(np.int64),
        "creatinine_phosphokinase":np.random.randint(50, 3000, n).astype(np.int64),
        "diabetes":                np.random.randint(0, 2, n).astype(np.int64),
        "ejection_fraction":       np.random.randint(15, 75, n).astype(np.int64),
        "high_blood_pressure":     np.random.randint(0, 2, n).astype(np.int64),
        "platelets":               np.random.uniform(100000, 500000, n).astype(np.float64),
        "serum_creatinine":        np.random.uniform(0.5, 4.0, n).astype(np.float64),
        "serum_sodium":            np.random.randint(125, 145, n).astype(np.int64),
        "sex":                     np.random.randint(0, 2, n).astype(np.int64),
        "smoking":                 np.random.randint(0, 2, n).astype(np.int64),
        "time":                    np.random.randint(4, 285, n).astype(np.int64),
        "DEATH_EVENT":             np.array([0]*35 + [1]*15, dtype=np.int64),
    })


# ── TEST 1: load_data ──────────────────────────────────────────────────────
class TestLoadData(unittest.TestCase):

    def test_shape(self):
        """Vérifie 299 lignes et 13 colonnes"""
        df = load_data(DATA_PATH)
        self.assertEqual(df.shape, (299, 13))

    def test_target_column_exists(self):
        """Vérifie que DEATH_EVENT existe"""
        df = load_data(DATA_PATH)
        self.assertIn("DEATH_EVENT", df.columns)

    def test_returns_dataframe(self):
        """Vérifie que c'est bien un DataFrame"""
        df = load_data(DATA_PATH)
        self.assertIsInstance(df, pd.DataFrame)


# ── TEST 2: missing values ─────────────────────────────────────────────────
class TestMissingValues(unittest.TestCase):

    def test_no_missing_in_raw_data(self):
        """La vraie data n'a pas de valeurs manquantes"""
        df = load_data(DATA_PATH)
        self.assertEqual(df.isnull().sum().sum(), 0)

    def test_fills_numerical_nulls(self):
        """Vérifie que les NaN numériques sont remplis"""
        df = make_sample_df()
        df.loc[0, "age"] = np.nan
        df.loc[1, "serum_creatinine"] = np.nan
        df_filled = handle_missing_values(df)
        self.assertEqual(df_filled.isnull().sum().sum(), 0)

    def test_fills_binary_nulls(self):
        """Vérifie que les NaN binaires sont remplis"""
        df = make_sample_df()
        df.loc[3, "anaemia"] = np.nan
        df_filled = handle_missing_values(df)
        self.assertEqual(df_filled["anaemia"].isnull().sum(), 0)

    def test_preserves_shape(self):
        """Vérifie que la shape ne change pas"""
        df = make_sample_df()
        df_filled = handle_missing_values(df)
        self.assertEqual(df_filled.shape, df.shape)


# ── TEST 3: optimize_memory ────────────────────────────────────────────────
class TestOptimizeMemory(unittest.TestCase):

    def test_memory_is_reduced(self):
        """Vérifie que la mémoire diminue"""
        df = load_data(DATA_PATH)
        before = df.memory_usage(deep=True).sum()
        df_opt = optimize_memory(df)
        after = df_opt.memory_usage(deep=True).sum()
        self.assertLess(after, before)

    def test_no_float64_columns(self):
        """Vérifie qu'il n'y a plus de float64"""
        df = load_data(DATA_PATH)
        df_opt = optimize_memory(df)
        cols = df_opt.select_dtypes(include=["float64"]).columns.tolist()
        self.assertEqual(cols, [])

    def test_no_int64_columns(self):
        """Vérifie qu'il n'y a plus de int64"""
        df = load_data(DATA_PATH)
        df_opt = optimize_memory(df)
        cols = df_opt.select_dtypes(include=["int64"]).columns.tolist()
        self.assertEqual(cols, [])

    def test_reduction_at_least_50_pct(self):
        """Vérifie réduction d'au moins 50%"""
        df = load_data(DATA_PATH)
        before = df.memory_usage(deep=True).sum()
        df_opt = optimize_memory(df)
        after = df_opt.memory_usage(deep=True).sum()
        reduction = (1 - after / before) * 100
        self.assertGreaterEqual(reduction, 50)


# ── TEST 4: outlier capping ────────────────────────────────────────────────
class TestCapOutliers(unittest.TestCase):

    def test_no_values_above_upper(self):
        """Vérifie qu'il n'y a plus de valeurs au dessus de la borne IQR"""
        df = load_data(DATA_PATH)
        df_capped = cap_outliers_iqr(df)
        for col in ["creatinine_phosphokinase", "platelets", "serum_creatinine"]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            upper = Q3 + 1.5 * (Q3 - Q1)
            self.assertTrue((df_capped[col] <= upper + 1e-6).all())

    def test_shape_preserved(self):
        """Vérifie que la shape ne change pas"""
        df = load_data(DATA_PATH)
        df_capped = cap_outliers_iqr(df)
        self.assertEqual(df_capped.shape, df.shape)


# ── TEST 5: feature engineering ───────────────────────────────────────────
class TestFeatureEngineering(unittest.TestCase):

    def test_new_features_created(self):
        """Vérifie que les 5 nouvelles features existent"""
        df = feature_engineering(make_sample_df())
        for col in ["age_group", "ef_severity", "sodium_risk",
                    "cp_log", "platelet_log"]:
            self.assertIn(col, df.columns)

    def test_sodium_risk_is_binary(self):
        """Vérifie que sodium_risk est 0 ou 1"""
        df = feature_engineering(make_sample_df())
        self.assertTrue(set(df["sodium_risk"].unique()).issubset({0, 1}))

    def test_ef_severity_range(self):
        """Vérifie que ef_severity est entre 0 et 3"""
        df = feature_engineering(make_sample_df())
        self.assertTrue(df["ef_severity"].between(0, 3).all())


# ── TEST 6: split data ────────────────────────────────────────────────────
class TestSplitData(unittest.TestCase):

    def test_split_sizes(self):
        """Vérifie que train + test = total"""
        df = load_data(DATA_PATH)
        X = df.drop("DEATH_EVENT", axis=1)
        y = df["DEATH_EVENT"]
        X_tr, X_te, y_tr, y_te = split_data(X, y, test_size=0.2)
        self.assertEqual(len(X_tr) + len(X_te), len(X))

    def test_test_size_60(self):
        """Vérifie que le test set = 60 lignes (20% de 299)"""
        df = load_data(DATA_PATH)
        X = df.drop("DEATH_EVENT", axis=1)
        y = df["DEATH_EVENT"]
        X_tr, X_te, y_tr, y_te = split_data(X, y, test_size=0.2)
        self.assertEqual(len(X_te), 60)

    def test_no_overlap(self):
        """Vérifie qu'il n'y a pas de données en commun"""
        df = make_sample_df()
        X = df.drop("DEATH_EVENT", axis=1)
        y = df["DEATH_EVENT"]
        X_tr, X_te, y_tr, y_te = split_data(X, y, test_size=0.2)
        self.assertEqual(len(set(X_tr.index) & set(X_te.index)), 0)


# ── TEST 7: class imbalance ───────────────────────────────────────────────
class TestHandleClassImbalance(unittest.TestCase):

    def test_oversample_balances(self):
        """Vérifie que les classes sont équilibrées après oversampling"""
        df = make_sample_df()
        X = df.drop("DEATH_EVENT", axis=1)
        y = df["DEATH_EVENT"]
        X_tr, X_te, y_tr, y_te = split_data(X, y, test_size=0.2)
        X_bal, y_bal = handle_class_imbalance(X_tr, y_tr)
        counts = y_bal.value_counts()
        self.assertEqual(counts[0], counts[1])

    def test_test_set_not_touched(self):
        """Vérifie que X_test n'est PAS modifié"""
        df = load_data(DATA_PATH)
        X = df.drop("DEATH_EVENT", axis=1)
        y = df["DEATH_EVENT"]
        X_tr, X_te, y_tr, y_te = split_data(X, y, test_size=0.2)
        size_before = len(X_te)
        X_bal, y_bal = handle_class_imbalance(X_tr, y_tr)
        self.assertEqual(len(X_te), size_before)


# ── TEST 8: full pipeline ─────────────────────────────────────────────────
class TestFullPipeline(unittest.TestCase):

    def test_all_keys_present(self):
        """Vérifie que le pipeline retourne toutes les clés"""
        result = run_full_pipeline(DATA_PATH)
        for key in ["X_train_scaled", "X_test_scaled",
                    "y_train", "y_test", "scaler", "feature_names"]:
            self.assertIn(key, result)

    def test_test_set_size(self):
        """Vérifie que X_test = 60 lignes"""
        result = run_full_pipeline(DATA_PATH)
        self.assertEqual(len(result["y_test"]), 60)

    def test_shapes_match(self):
        """Vérifie que train et test ont le même nombre de features"""
        result = run_full_pipeline(DATA_PATH)
        self.assertEqual(
            result["X_train_scaled"].shape[1],
            result["X_test_scaled"].shape[1]
        )

    def test_no_missing_after_pipeline(self):
        """Vérifie qu'il n'y a plus de NaN après le pipeline"""
        result = run_full_pipeline(DATA_PATH)
        self.assertEqual(
            pd.DataFrame(result["X_train_scaled"]).isnull().sum().sum(), 0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)