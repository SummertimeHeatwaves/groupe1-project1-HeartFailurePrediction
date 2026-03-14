import os
import pandas as pd

def test_data_file_exists():
    """Vérifie si le fichier source de la clinique existe."""
    # Remplacer par le chemin exact de votre CSV s'il est différent
    data_path = "data/data_processed.csv"
    assert os.path.exists(data_path), "Le dataset Heart Failure est introuvable."