import os
import subprocess
import sys
def test_shap_analysis_execution():
    """Vérifie que l'analyse SHAP génère bien les graphiques d'explication."""
    # Exécuter le script SHAP
    result = subprocess.run([sys.executable, "src/shap_analysis.py"], capture_output=True, text=True)
    
    assert result.returncode == 0
    # Vérifier que les deux images SHAP obligatoires sont présentes
    assert os.path.exists("notebooks/figures/shap_summary.png")
    assert os.path.exists("notebooks/figures/shap_bar.png")