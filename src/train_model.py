"""
train_model.py
--------------
Entraînement et sélection du meilleur modèle ML
pour la prédiction du risque de défaillance cardiaque.

Modèles entraînés :
    1. Random Forest Classifier
    2. XGBoost Classifier
    3. LightGBM Classifier
    4. Logistic Regression

Métriques d'évaluation :
    - ROC-AUC   (métrique principale)
    - Accuracy
    - Precision
    - Recall
    - F1-Score

Usage :
    python src/train_model.py
"""

# ══════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════
import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

import xgboost  as xgb
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Ajouter src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_processing import run_full_pipeline

# ══════════════════════════════════════════════
# CHEMINS
# ══════════════════════════════════════════════
DATA_PATH  = os.path.join(os.path.dirname(__file__), "../data/DATASET.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "../models")
FIGURE_DIR = os.path.join(os.path.dirname(__file__), "../notebooks/figures")
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ══════════════════════════════════════════════
# 1. DÉFINITION DES MODÈLES
# ══════════════════════════════════════════════
def get_models():
    """
    Retourne un dictionnaire avec les 4 modèles
    et leurs hyperparamètres.
    """
    models = {

        # ── 1. Random Forest ─────────────────
        "Random Forest": RandomForestClassifier(
            n_estimators      = 100,
            max_depth         = 10,
            min_samples_split = 5,
            class_weight      = "balanced",
            random_state      = 42,
            n_jobs            = -1
        ),

        # ── 2. XGBoost ───────────────────────
        "XGBoost": xgb.XGBClassifier(
            n_estimators     = 100,
            max_depth        = 6,
            learning_rate    = 0.1,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            eval_metric      = "logloss",
            random_state     = 42,
            verbosity        = 0
        ),

        # ── 3. LightGBM ──────────────────────
        "LightGBM": lgb.LGBMClassifier(
            n_estimators  = 100,
            max_depth     = 6,
            learning_rate = 0.1,
            num_leaves    = 31,
            class_weight  = "balanced",
            random_state  = 42,
            verbose       = -1
        ),

        # ── 4. Logistic Regression ───────────
        "Logistic Regression": LogisticRegression(
            max_iter     = 1000,
            class_weight = "balanced",
            solver       = "lbfgs",
            random_state = 42
        ),
    }
    return models


# ══════════════════════════════════════════════
# 2. ENTRAÎNEMENT D'UN MODÈLE
# ══════════════════════════════════════════════
def train_model(name, model, X_train, y_train):
    """Entraîne un modèle et retourne le modèle entraîné."""
    print(f"\n  Entraînement : {name}...")
    model.fit(X_train, y_train)
    print(f"  {name} entraine !")
    return model


# ══════════════════════════════════════════════
# 3. ÉVALUATION D'UN MODÈLE
# ══════════════════════════════════════════════
def evaluate_model(name, model, X_test, y_test):
    """
    Calcule toutes les métriques :
    ROC-AUC, Accuracy, Precision, Recall, F1-Score
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Modele"    : name,
        "ROC-AUC"   : round(roc_auc_score(y_test, y_proba), 4),
        "Accuracy"  : round(accuracy_score(y_test, y_pred),  4),
        "Precision" : round(precision_score(y_test, y_pred), 4),
        "Recall"    : round(recall_score(y_test, y_pred),    4),
        "F1-Score"  : round(f1_score(y_test, y_pred),        4),
    }

    print(f"\n  {'─'*45}")
    print(f"  {name}")
    print(f"  {'─'*45}")
    print(f"  ROC-AUC   : {metrics['ROC-AUC']}")
    print(f"  Accuracy  : {metrics['Accuracy']}")
    print(f"  Precision : {metrics['Precision']}")
    print(f"  Recall    : {metrics['Recall']}")
    print(f"  F1-Score  : {metrics['F1-Score']}")
    print(f"\n  Rapport detaille :")
    print(classification_report(
        y_test, y_pred,
        target_names=["Survivant (0)", "Decede (1)"]
    ))
    print(f"  Matrice de confusion :")
    print(f"  {confusion_matrix(y_test, y_pred)}")

    return metrics, y_pred, y_proba


# ══════════════════════════════════════════════
# 4. GRAPHIQUES
# ══════════════════════════════════════════════
def plot_roc_curves(models_dict, X_test, y_test):
    """Courbes ROC pour tous les modèles."""
    plt.figure(figsize=(10, 7))
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

    for (name, model), color in zip(models_dict.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, color=color,
                 label=f"{name} (AUC = {auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Aleatoire")
    plt.xlabel("Taux de Faux Positifs", fontsize=12)
    plt.ylabel("Taux de Vrais Positifs", fontsize=12)
    plt.title("Courbes ROC - Comparaison des Modeles",
              fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "roc_curves.png")
    plt.savefig(path)
    plt.close()
    print(f"  [plot] Courbes ROC sauvegardees")


def plot_metrics_comparison(df_metrics):
    """Graphique comparaison des métriques."""
    metrics_cols = ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1-Score"]
    df_plot = df_metrics.set_index("Modele")[metrics_cols]

    ax = df_plot.plot(kind="bar", figsize=(12, 6),
                      colormap="Set2", edgecolor="black")
    ax.set_title("Comparaison des Metriques - 4 Modeles",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "metrics_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"  [plot] Comparaison metriques sauvegardee")


def plot_confusion_matrices(models_dict, X_test, y_test):
    """Matrices de confusion des 4 modèles."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (name, model) in enumerate(models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        axes[i].imshow(cm, cmap="Blues")
        axes[i].set_title(name, fontweight="bold")
        axes[i].set_xlabel("Predit")
        axes[i].set_ylabel("Reel")
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xticklabels(["Survivant", "Decede"])
        axes[i].set_yticklabels(["Survivant", "Decede"])
        for r in range(2):
            for c in range(2):
                axes[i].text(c, r, cm[r, c],
                             ha="center", va="center",
                             fontsize=16, fontweight="bold",
                             color="white" if cm[r, c] > cm.max()/2
                             else "black")

    plt.suptitle("Matrices de Confusion - 4 Modeles",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "confusion_matrices.png")
    plt.savefig(path)
    plt.close()
    print(f"  [plot] Matrices de confusion sauvegardees")


# ══════════════════════════════════════════════
# 5. SÉLECTION DU MEILLEUR MODÈLE
# ══════════════════════════════════════════════
def select_best_model(all_metrics, all_models):
    """
    Selectionne le meilleur modele base sur ROC-AUC.

    Justification :
    ROC-AUC est la metrique principale car :
    - Dataset desequilibre (68/32)
    - Mesure la discrimination independamment du seuil
    - Critique en medecine : minimiser les faux negatifs
    """
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics = df_metrics.sort_values("ROC-AUC", ascending=False)
    df_metrics = df_metrics.reset_index(drop=True)

    print("\n" + "="*55)
    print("  COMPARAISON FINALE DES 4 MODELES")
    print("="*55)
    print(df_metrics.to_string(index=False))

    best_name  = df_metrics.iloc[0]["Modele"]
    best_auc   = df_metrics.iloc[0]["ROC-AUC"]
    best_model = all_models[best_name]

    print(f"\n{'='*55}")
    print(f"  MEILLEUR MODELE : {best_name}")
    print(f"  ROC-AUC         : {best_auc}")
    print(f"{'='*55}")

    return best_name, best_model, df_metrics


# ══════════════════════════════════════════════
# 6. SAUVEGARDE
# ══════════════════════════════════════════════
def save_all(all_models, best_name, scaler, feature_names, df_metrics):
    """Sauvegarde tous les modeles et artefacts."""
    print("\n  Sauvegarde des fichiers...")

    for name, model in all_models.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, os.path.join(MODEL_DIR, filename))
        print(f"  [save] {name} -> {filename}")

    joblib.dump(all_models[best_name],
                os.path.join(MODEL_DIR, "best_model.pkl"))
    print(f"  [save] Meilleur modele ({best_name}) -> best_model.pkl")

    joblib.dump(scaler,
                os.path.join(MODEL_DIR, "scaler.pkl"))
    print(f"  [save] Scaler -> scaler.pkl")

    joblib.dump(feature_names,
                os.path.join(MODEL_DIR, "feature_names.pkl"))
    print(f"  [save] Feature names -> feature_names.pkl")

    joblib.dump({"best_model_name": best_name,
                 "feature_names": feature_names},
                os.path.join(MODEL_DIR, "model_info.pkl"))

    df_metrics.to_csv(os.path.join(MODEL_DIR, "metrics.csv"), index=False)
    print(f"  [save] Metriques -> metrics.csv")


# ══════════════════════════════════════════════
# 7. PIPELINE COMPLET
# ══════════════════════════════════════════════
def run_training_pipeline():
    """
    Pipeline complet d'entrainement :
    1. Preparer la data
    2. Definir les 4 modeles
    3. Entrainer les 4 modeles
    4. Evaluer avec toutes les metriques
    5. Choisir le meilleur modele
    6. Generer les graphiques
    7. Sauvegarder tout
    """
    print("\n" + "="*55)
    print("  HEART FAILURE - TRAINING PIPELINE")
    print("="*55)

    # ETAPE 1 : Preparer la data
    print("\n[ETAPE 1/7] Preparation de la data...")
    result = run_full_pipeline(DATA_PATH)
    X_train       = result["X_train_scaled"]
    X_test        = result["X_test_scaled"]
    y_train       = result["y_train"]
    y_test        = result["y_test"]
    scaler        = result["scaler"]
    feature_names = result["feature_names"]
    print(f"\n  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_train : {dict(y_train.value_counts())}")
    print(f"  y_test  : {dict(y_test.value_counts())}")

    # ETAPE 2 : Definir les modeles
    print("\n[ETAPE 2/7] Definition des 4 modeles...")
    models = get_models()
    for name in models:
        print(f"  -> {name}")

    # ETAPE 3 : Entrainer
    print("\n[ETAPE 3/7] Entrainement des modeles...")
    trained_models = {}
    for name, model in models.items():
        trained_models[name] = train_model(
            name, model, X_train, y_train)

    # ETAPE 4 : Evaluer
    print("\n[ETAPE 4/7] Evaluation des modeles...")
    all_metrics = []
    for name, model in trained_models.items():
        metrics, _, _ = evaluate_model(
            name, model, X_test, y_test)
        all_metrics.append(metrics)

    # ETAPE 5 : Meilleur modele
    print("\n[ETAPE 5/7] Selection du meilleur modele...")
    best_name, best_model, df_metrics = select_best_model(
        all_metrics, trained_models)

    # ETAPE 6 : Graphiques
    print("\n[ETAPE 6/7] Generation des graphiques...")
    plot_roc_curves(trained_models, X_test, y_test)
    plot_metrics_comparison(df_metrics)
    plot_confusion_matrices(trained_models, X_test, y_test)

    # ETAPE 7 : Sauvegarder
    print("\n[ETAPE 7/7] Sauvegarde...")
    save_all(trained_models, best_name,
             scaler, feature_names, df_metrics)

    # RESUME FINAL
    print("\n" + "="*55)
    print("  TRAINING PIPELINE COMPLET !")
    print("="*55)
    print(f"\n  Meilleur modele : {best_name}")
    print(f"\n  Fichiers dans models/ :")
    print(f"    -> best_model.pkl")
    print(f"    -> scaler.pkl")
    print(f"    -> metrics.csv")
    print(f"\n  Graphiques dans notebooks/figures/ :")
    print(f"    -> roc_curves.png")
    print(f"    -> metrics_comparison.png")
    print(f"    -> confusion_matrices.png")
    print("="*55)

    return best_name, best_model, df_metrics


# ══════════════════════════════════════════════
# LANCER
# ══════════════════════════════════════════════
if __name__ == "__main__":
    best_name, best_model, df_metrics = run_training_pipeline()
