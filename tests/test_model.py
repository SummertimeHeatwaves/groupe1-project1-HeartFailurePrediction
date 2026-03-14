import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def trained_model():
    """Create and train a dummy model for testing."""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(200, 5),
        columns=[f"feature_{i}" for i in range(5)]
    )
    y = pd.Series(np.random.choice([0, 1], size=200, p=[0.68, 0.32]))
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model, X, y

class TestModelPrediction:
    """Tests for model prediction functionality."""
    
    def test_prediction_output_length(self, trained_model):
        model, X, _ = trained_model
        preds = model.predict(X)
        assert len(preds) == len(X)
        
    def test_prediction_binary_values(self, trained_model):
        model, X, _ = trained_model
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})
        
    def test_probability_range_0_to_1(self, trained_model):
        model, X, _ = trained_model
        proba = model.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        
    def test_probabilities_sum_to_one(self, trained_model):
        model, X, _ = trained_model
        proba = model.predict_proba(X)
        sums = proba.sum(axis=1)
        np.testing.assert_array_almost_equal(sums, 1.0, decimal=10)
        
    def test_single_sample_prediction(self, trained_model):
        model, X, _ = trained_model
        single = X.iloc[[0]]
        pred = model.predict(single)
        assert len(pred) == 1
        assert pred[0] in [0, 1]
        
    def test_predict_proba_shape(self, trained_model):
        model, X, _ = trained_model
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        