"""
task3_models.py - Train LightGBM Classifier and Logistic Regression model.

Ce script entraîne et compare les deux modèles sur le dataset
Heart Failure avec les métriques complètes.

"""

import pandas as pd
import numpy as np
import warnings
import os
import sys
import joblib

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import du pipeline de preprocessing (Membre 1)
from src.data_processing import run_preprocessing_pipeline


# ================================================================
# ÉTAPE 1 : Définir les modèles
# ================================================================

def get_models() -> dict:
    """Retourne les deux modèles configurés avec leurs hyperparamètres."""
    return {
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            num_leaves=31,
            min_child_samples=10,
            class_weight="balanced",   # Gère le déséquilibre des classes
            random_state=42,
            verbose=-1                 # Silencieux
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0,                     # Régularisation L2
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",   # Gère le déséquilibre des classes
            random_state=42
        )
    }


# ================================================================
# ÉTAPE 2 : Évaluer un modèle sur le test set
# ================================================================

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Calcule toutes les métriques sur le test set."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name" : model_name,
        "accuracy"   : round(accuracy_score(y_test, y_pred), 4),
        "precision"  : round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall"     : round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score"   : round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc"    : round(roc_auc_score(y_test, y_proba), 4),
    }

    print(f"\n{'='*50}")
    print(f"  Modèle : {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}  ← critique en médecine !")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}  ← métrique principale")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Survived','Deceased'])}")

    return metrics


# ================================================================
# ÉTAPE 3 : Cross-validation
# ================================================================

def cross_validate(model, X_train, y_train, model_name: str) -> float:
    """Validation croisée StratifiedKFold (k=5) — ROC-AUC."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X_train, y_train,
        cv=skf, scoring="roc_auc"
    )
    print(f"  CV ROC-AUC ({model_name}): {scores.mean():.4f} (+/- {scores.std():.4f})")
    return round(scores.mean(), 4)


# ================================================================
# ÉTAPE 4 : Visualisations
# ================================================================

def plot_confusion_matrices(models_dict, X_test, y_test):
    """Affiche les matrices de confusion côte à côte."""
    os.makedirs("reports/figures", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (name, model) in zip(axes, models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Survived", "Deceased"],
            yticklabels=["Survived", "Deceased"],
            annot_kws={"size": 14}
        )
        ax.set_title(f"Confusion Matrix\n{name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)

    plt.suptitle("LightGBM vs Logistic Regression", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrices_task3.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n✅ Confusion matrices sauvegardées dans reports/figures/")


def plot_comparison_bar(results: list):
    """Graphique comparatif des métriques entre les deux modèles."""
    os.makedirs("reports/figures", exist_ok=True)
    metrics  = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    labels   = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x        = np.arange(len(metrics))
    width    = 0.35
    colors   = ["#3B82F6", "#10B981"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (res, color) in enumerate(zip(results, colors)):
        vals = [res[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=res["model_name"],
                      color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("LightGBM vs Logistic Regression — Comparaison des métriques",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/figures/comparison_task3.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Graphique comparatif sauvegardé dans reports/figures/")


# ================================================================
# ÉTAPE 5 : Sauvegarder le meilleur modèle
# ================================================================

def save_best_model(models_dict: dict, results: list, scaler, feature_names: list):
    """Sauvegarde le meilleur modèle selon le ROC-AUC."""
    os.makedirs("models", exist_ok=True)
    best = max(results, key=lambda x: x["roc_auc"])
    best_model = models_dict[best["model_name"]]

    joblib.dump(best_model,    "models/best_model.pkl")
    joblib.dump(scaler,        "models/scaler.pkl")
    joblib.dump(feature_names, "models/feature_names.pkl")

    print(f"\n✅ Meilleur modèle sauvegardé : {best['model_name']}")
    print(f"   ROC-AUC : {best['roc_auc']:.4f}")
    print(f"   Fichiers : models/best_model.pkl, models/scaler.pkl")
    return best


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    # 1. Preprocessing (Membre 1)
    print("\n" + "="*50)
    print("  ÉTAPE 1 : Preprocessing des données")
    print("="*50)
    data = run_preprocessing_pipeline()
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]

    # 2. Entraînement
    print("\n" + "="*50)
    print("  ÉTAPE 2 : Entraînement des modèles")
    print("="*50)
    models  = get_models()
    results = []
    trained = {}

    for name, model in models.items():
        print(f"\n--- Training : {name} ---")
        model.fit(X_train, y_train)
        trained[name] = model

        # Cross-validation
        cv_auc = cross_validate(model, X_train, y_train, name)

        # Évaluation sur test set
        metrics = evaluate_model(model, X_test, y_test, name)
        metrics["cv_roc_auc"] = cv_auc
        results.append(metrics)

    # 3. Tableau récapitulatif
    print("\n" + "="*65)
    print(f"{'Modèle':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("="*65)
    for r in results:
        best_mark = " ⭐" if r == max(results, key=lambda x: x["roc_auc"]) else ""
        print(
            f"{r['model_name']:<25} "
            f"{r['accuracy']:>9.4f} "
            f"{r['precision']:>10.4f} "
            f"{r['recall']:>8.4f} "
            f"{r['f1_score']:>8.4f} "
            f"{r['roc_auc']:>8.4f}{best_mark}"
        )
    print("="*65)

    # 4. Visualisations
    print("\n" + "="*50)
    print("  ÉTAPE 3 : Génération des visualisations")
    print("="*50)
    plot_confusion_matrices(trained, X_test, y_test)
    plot_comparison_bar(results)

    # 5. Sauvegarde du meilleur modèle
    print("\n" + "="*50)
    print("  ÉTAPE 4 : Sauvegarde du meilleur modèle")
    print("="*50)
    best = save_best_model(trained, results, data["scaler"], data["feature_names"])

    print(f"\n🎉 Tâche terminée ! Meilleur modèle : {best['model_name']} (AUC={best['roc_auc']:.4f})")
