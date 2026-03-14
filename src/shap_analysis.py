<<<<<<< Updated upstream
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("models/best_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")
df = pd.read_csv("data/data_processed.csv")

X = df.drop("DEATH_EVENT", axis=1)[feature_names]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, show=False)
plt.savefig("notebooks/figures/shap_summary.png", bbox_inches="tight")
plt.close()

shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.savefig("notebooks/figures/shap_bar.png", bbox_inches="tight")
plt.close()

print("SHAP analysis terminée !")
=======
import os
import sys
import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIGURES_DIR = "reports/figures"

def load_artifacts(model_dir: str = "models") -> tuple:
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    features = joblib.load(os.path.join(model_dir, "feature_names.pkl"))
    logger.info("Artifacts loaded for SHAP analysis")
    return model, scaler, features

def compute_shap_values(model, X_data: pd.DataFrame):
    logger.info(f"Computing SHAP values for {X_data.shape[0]} samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_data)
    logger.info("SHAP values computed successfully")
    return shap_values

def plot_summary_beeswarm(shap_values, X_data: pd.DataFrame, save_path: Optional[str] = None):
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "shap_summary_beeswarm.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_data, show=False, plot_size=None)
    plt.title("SHAP Summary: Feature Impact on Heart Failure Risk", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("SHAP Value (impact on model output)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Beeswarm plot saved: {save_path}")

def plot_bar_importance(shap_values, save_path: Optional[str] = None):
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "shap_bar_importance.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.title("Feature Importance (Mean |SHAP| Value)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Bar importance plot saved: {save_path}")

def plot_dependence(shap_values, X_data: pd.DataFrame, feature: str = "serum_creatinine", interaction_feature: Optional[str] = None, save_path: Optional[str] = None):
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, f"shap_dep_{feature}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature, shap_values.values, X_data, interaction_index=interaction_feature, show=False)
    plt.title(f"SHAP Dependence: {feature}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Dependence plot saved: {save_path}")

def plot_waterfall_patient(shap_values, index: int = 0, save_path: Optional[str] = None):
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, f"shap_waterfall_patient_{index}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_values[index], show=False)
    plt.title(f"SHAP Explanation - Patient #{index}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Waterfall plot saved: {save_path}")

def generate_all_shap_plots(model, X_test: pd.DataFrame, feature_names: list):
    logger.info("Generating all SHAP plots...")
    shap_values = compute_shap_values(model, X_test)
    plot_summary_beeswarm(shap_values, X_test)
    plot_bar_importance(shap_values)
    for feat in ["serum_creatinine", "ejection_fraction", "age"]:
        if feat in X_test.columns:
            plot_dependence(shap_values, X_test, feature=feat)
    for i in range(min(3, len(X_test))):
        plot_waterfall_patient(shap_values, index=i)
    
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    
    logger.info("\nTop Features by Mean |SHAP|:")
    for _, row in feature_importance.iterrows():
        logger.info(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")
    logger.info("All SHAP plots generated successfully!")
    return shap_values

if __name__ == "__main__":
    from src.data_processing import run_preprocessing_pipeline
    data = run_preprocessing_pipeline()
    model, scaler, features = load_artifacts()
    shap_values = generate_all_shap_plots(model, data["X_test"], features)
>>>>>>> Stashed changes
