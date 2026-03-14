import os

def test_model_generated():
    """Vérifie si l'entraînement a bien généré le modèle."""
    assert os.path.exists("models/best_model.pkl"), "Le modèle Random Forest n'a pas été généré."
    assert os.path.exists("models/scaler.pkl"), "Le scaler n'a pas été généré."