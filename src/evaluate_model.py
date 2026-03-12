import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve
)
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIGURES_DIR = "reports/figures"

def load_model(model_dir: str = "models") -> tuple:
    """Load all saved model artifacts."""
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    features = joblib.load(os.path.join(model_dir, "feature_names.pkl"))
    logger.info(f"Model artifacts loaded from {model_dir}/")
    return model, scaler, features

def plot_confusion_matrix(y_true, y_pred, save_path: str = None):
    """Generate and save a styled confusion matrix plot."""
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Survived", "Deceased"],
        yticklabels=["Survived", "Deceased"],
        ax=ax, annot_kws={"size": 16}
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title("Confusion Matrix - Best Model", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved: {save_path}")

def plot_roc_curve(y_true, y_proba, save_path: str = None):
    """Generate and save ROC curve with AUC annotation."""
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "roc_curve.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2E86C1", lw=2.5, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2E86C1")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curve saved: {save_path}")

def plot_precision_recall_curve(y_true, y_proba, save_path: str = None):
    """Generate Precision-Recall curve."""
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "precision_recall_curve.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#E74C3C", lw=2.5)
    ax.fill_between(recall, precision, alpha=0.1, color="#E74C3C")
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"PR curve saved: {save_path}")

if __name__ == "__main__":
    from src.data_processing import run_preprocessing_pipeline
    data = run_preprocessing_pipeline()
    model, scaler, features = load_model()
    
    y_pred = model.predict(data["X_test"])
    y_proba = model.predict_proba(data["X_test"])[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(data["y_test"], y_pred, target_names=["Survived", "Deceased"]))
    
    plot_roc_curve(data["y_test"], y_proba)
    plot_confusion_matrix(data["y_test"], y_pred)
    plot_precision_recall_curve(data["y_test"], y_proba)
    print("All evaluation plots generated.")