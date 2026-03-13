import os
import subprocess
import sys

def test_evaluate_model_execution():
    """Vérifie que l'évaluation génère les graphiques de performance."""
    # On force l'environnement en UTF-8 pour éviter les erreurs de caractères
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    # On s'assure d'abord que le modèle existe
    if not os.path.exists("models/best_model.pkl"):
        subprocess.run([sys.executable, "src/train_model.py"], env=env)

    # On lance l'évaluation
    # Note : on utilise errors='replace' pour ne pas planter sur un caractère spécial
    result = subprocess.run(
        [sys.executable, "src/evaluate_model.py"], 
        capture_output=True, 
        text=True, 
        env=env,
        encoding="utf-8",
        errors="replace" 
    )

    # Si ça échoue, on affiche l'erreur réelle du script pour debug
    if result.returncode != 0:
        print("\n--- ERREUR DANS LE SCRIPT EVALUATE_MODEL ---")
        print(result.stderr)
        print("--------------------------------------------")

    assert result.returncode == 0
    assert os.path.exists("notebooks/figures/evaluation_finale.png")