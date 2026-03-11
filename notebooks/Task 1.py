"""
test_data_processing.py - Unit tests for data_processing module.

Tests cover: memory optimization, missing values detection,
outlier detection, and class distribution analysis.

Run with: pytest tests/test_data_processing.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_processing import (
    optimize_memory,
    check_missing_values,
    detect_outliers_iqr,
    analyze_class_distribution,
    handle_outliers
)


# ================================================================
# FIXTURE — données de test réalistes
# ================================================================

@pytest.fixture
def sample_df():
    """Create a realistic sample DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "age":                    np.random.randint(40, 95, size=n).astype(np.int64),
        "anaemia":                np.random.choice([0, 1], size=n).astype(np.int64),
        "creatinine_phosphokinase": np.random.randint(23, 7861, size=n).astype(np.int64),
        "diabetes":               np.random.choice([0, 1], size=n).astype(np.int64),
        "ejection_fraction":      np.random.randint(14, 80, size=n).astype(np.int64),
        "high_blood_pressure":    np.random.choice([0, 1], size=n).astype(np.int64),
        "platelets":              np.random.uniform(25000, 850000, size=n).astype(np.float64),
        "serum_creatinine":       np.random.uniform(0.5, 9.4, size=n).astype(np.float64),
        "serum_sodium":           np.random.randint(113, 148, size=n).astype(np.int64),
        "sex":                    np.random.choice([0, 1], size=n).astype(np.int64),
        "smoking":                np.random.choice([0, 1], size=n).astype(np.int64),
        "DEATH_EVENT":            np.random.choice([0, 1], size=n, p=[0.68, 0.32]).astype(np.int64),
    })


# ================================================================
# TESTS — optimize_memory()
# ================================================================

class TestOptimizeMemory:
    """Tests for the optimize_memory function."""

    def test_reduces_memory_usage(self, sample_df):
        """La mémoire après optimisation doit être strictement inférieure."""
        mem_before = sample_df.memory_usage(deep=True).sum()
        df_opt = optimize_memory(sample_df.copy())
        mem_after = df_opt.memory_usage(deep=True).sum()
        assert mem_after < mem_before, \
            f"Expected reduction: {mem_before} > {mem_after}"

    def test_preserves_values_float(self, sample_df):
        """Les valeurs float doivent être conservées à la précision float32."""
        df_opt = optimize_memory(sample_df.copy())
        np.testing.assert_array_almost_equal(
            df_opt["serum_creatinine"].values,
            sample_df["serum_creatinine"].values.astype(np.float32),
            decimal=5
        )

    def test_preserves_values_int(self, sample_df):
        """Les valeurs entières doivent être exactement conservées."""
        df_opt = optimize_memory(sample_df.copy())
        np.testing.assert_array_equal(
            df_opt["age"].values,
            sample_df["age"].values
        )

    def test_returns_dataframe(self, sample_df):
        """Le type de retour doit être un DataFrame pandas."""
        result = optimize_memory(sample_df.copy())
        assert isinstance(result, pd.DataFrame)

    def test_preserves_shape(self, sample_df):
        """Le shape ne doit pas changer après optimisation."""
        result = optimize_memory(sample_df.copy())
        assert result.shape == sample_df.shape

    def test_float_columns_are_float32(self, sample_df):
        """Les colonnes float64 doivent devenir float32."""
        df_opt = optimize_memory(sample_df.copy())
        for col in ["platelets", "serum_creatinine"]:
            assert df_opt[col].dtype == np.float32, \
                f"{col} should be float32, got {df_opt[col].dtype}"


# ================================================================
# TESTS — check_missing_values()
# ================================================================

class TestCheckMissingValues:
    """Tests for check_missing_values function."""

    def test_no_missing_values(self, sample_df):
        """Doit retourner zéro pour un DataFrame complet."""
        result = check_missing_values(sample_df)
        assert result.sum() == 0

    def test_detects_missing(self, sample_df):
        """Doit détecter correctement les NaN introduits manuellement."""
        df = sample_df.copy()
        df.loc[0, "age"] = np.nan
        df.loc[5, "serum_creatinine"] = np.nan
        df.loc[10, "serum_creatinine"] = np.nan
        result = check_missing_values(df)
        assert result["age"] == 1
        assert result["serum_creatinine"] == 2


# ================================================================
# TESTS — detect_outliers_iqr()
# ================================================================

class TestDetectOutliers:
    """Tests for detect_outliers_iqr function."""

    def test_returns_boolean_mask(self, sample_df):
        """Le résultat doit être un masque booléen."""
        cols = ["age", "serum_creatinine"]
        result = detect_outliers_iqr(sample_df, columns=cols)
        assert result.dtypes["age"] == bool

    def test_detects_extreme_values(self):
        """Doit détecter les valeurs aberrantes évidentes."""
        df = pd.DataFrame({"val": [1, 2, 3, 3, 4, 5, 200]})
        mask = detect_outliers_iqr(df, columns=["val"])
        assert mask["val"].iloc[-1] == True

    def test_no_outliers_in_tight_data(self):
        """Ne doit pas trouver d'outliers dans des données uniformes."""
        df = pd.DataFrame({"val": [5, 5, 5, 5, 5]})
        mask = detect_outliers_iqr(df, columns=["val"])
        assert mask["val"].sum() == 0


# ================================================================
# TESTS — analyze_class_distribution()
# ================================================================

class TestClassDistribution:
    """Tests for analyze_class_distribution function."""

    def test_returns_dict(self, sample_df):
        result = analyze_class_distribution(sample_df)
        assert isinstance(result, dict)
        assert "counts" in result
        assert "percentages" in result

    def test_contains_both_classes(self, sample_df):
        result = analyze_class_distribution(sample_df)
        assert 0 in result["counts"]
        assert 1 in result["counts"]

    def test_percentages_sum_to_100(self, sample_df):
        result = analyze_class_distribution(sample_df)
        total = sum(result["percentages"].values())
        assert abs(total - 100.0) < 0.5


# ================================================================
# TESTS — handle_outliers()
# ================================================================

class TestHandleOutliers:
    """Tests for handle_outliers function."""

    def test_preserves_shape(self, sample_df):
        result = handle_outliers(sample_df.copy())
        assert result.shape == sample_df.shape

    def test_clips_extreme_values(self):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5, 100]})
        result = handle_outliers(df, columns=["val"])
        assert result["val"].max() < 100
