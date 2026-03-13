import os
import subprocess
import sys

def test_train_model_execution():
    # Crée une copie de l'environnement actuel et ajoute la variable PYTHONUTF8
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1" 

    # Ajoute env=env dans subprocess.run
    result = subprocess.run(
        [sys.executable, "src/train_model.py"], 
        capture_output=True, 
        text=True, 
        env=env,
        encoding="utf-8" # Force aussi l'encodage de la lecture du résultat
    )
    
    assert result.returncode == 0
