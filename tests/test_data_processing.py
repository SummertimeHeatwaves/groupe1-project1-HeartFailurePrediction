import pytest
import pandas as pd
import numpy as np
import sys
import os

# Permet d'importer depuis src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_processing import (
    optimize_memory,
    check_missing_values,
    analyze_class_distribution,
    handle_outliers
)

@pytest.fixture
def sample_df():
    """Crée un DataFrame de test minimaliste."""
    np.random.seed(42)
    return pd.DataFrame({
        "age": [40, 50, 60, 70, 80],
        "platelets": [200000.0, 250000.0, 300000.0, 350000.0, 400000.0],
        "serum_creatinine": [1.0, 1.1, 1.2, 1.3, 9.0], # 9.0 est un outlier
        "DEATH_EVENT": [0, 0, 1, 1, 0]
    })

class TestDataPipeline:
    def test_optimize_memory_reduction(self, sample_df):
        """Vérifie que l'optimisation réduit bien la taille en mémoire."""
        mem_before = sample_df.memory_usage(deep=True).sum()
        df_opt = optimize_memory(sample_df.copy())
        mem_after = df_opt.memory_usage(deep=True).sum()
        assert mem_after < mem_before

    def test_check_missing_values(self, sample_df):
        """Vérifie la détection des valeurs manquantes."""
        result = check_missing_values(sample_df)
        assert result.sum() == 0

    def test_handle_outliers_clipping(self, sample_df):
        """Vérifie que les valeurs aberrantes sont bien plafonnées."""
        df_clean = handle_outliers(sample_df.copy(), columns=["serum_creatinine"])
        assert df_clean["serum_creatinine"].max() < 9.0

    def test_analyze_distribution_format(self, sample_df):
        """Vérifie que l'analyse de distribution retourne bien un dictionnaire."""
        result = analyze_class_distribution(sample_df)
        assert "counts" in result