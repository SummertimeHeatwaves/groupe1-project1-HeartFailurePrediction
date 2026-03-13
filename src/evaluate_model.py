"""
evaluate_model.py
-----------------
Evaluation complete du meilleur modele ML avec SHAP.

Contenu :
    1. Charger le meilleur modele
    2. Evaluer sur le test set
    3. Analyse SHAP (explainability)
    4. Graphiques SHAP
    5. Rapport final

Usage :
    python src/evaluate_model.py
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

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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
os.makedirs(FIGURE_DIR, exist_ok=True)


# ══════════════════════════════════════════════
# 1. CHARGER LE MEILLEUR MODELE
# ══════════════════════════════════════════════
def load_best_model():
    """
    Charge le meilleur modele, le scaler
    et les feature names depuis models/
    """
    print("\n[ETAPE 1/6] Chargement du meilleur modele...")

    best_model    = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    model_info    = joblib.load(os.path.join(MODEL_DIR, "model_info.pkl"))

    best_name = model_info["best_model_name"]
    print(f"  Meilleur modele charge : {best_name}")
    print(f"  Nombre de features     : {len(feature_names)}")

    return best_model, scaler, feature_names, best_name


# ══════════════════════════════════════════════
# 2. EVALUER LE MODELE
# ══════════════════════════════════════════════
def evaluate_best_model(model, X_test, y_test, feature_names):
    """
    Calcule toutes les metriques sur le test set.
    """
    print("\n[ETAPE 2/6] Evaluation du meilleur modele...")

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "ROC-AUC"          : round(roc_auc_score(y_test, y_proba),           4),
        "Accuracy"         : round(accuracy_score(y_test, y_pred),            4),
        "Precision"        : round(precision_score(y_test, y_pred),           4),
        "Recall"           : round(recall_score(y_test, y_pred),              4),
        "F1-Score"         : round(f1_score(y_test, y_pred),                  4),
        "Average Precision": round(average_precision_score(y_test, y_proba),  4),
    }

    print(f"\n  {'─'*45}")
    print(f"  METRIQUES FINALES SUR LE TEST SET")
    print(f"  {'─'*45}")
    for key, val in metrics.items():
        print(f"  {key:<20} : {val}")

    print(f"\n  Rapport detaille :")
    print(classification_report(
        y_test, y_pred,
        target_names=["Survivant (0)", "Decede (1)"]
    ))

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Matrice de confusion :")
    print(f"  TN={cm[0,0]} FP={cm[0,1]}")
    print(f"  FN={cm[1,0]} TP={cm[1,1]}")

    return metrics, y_pred, y_proba


# ══════════════════════════════════════════════
# 3. GRAPHIQUES D'EVALUATION
# ══════════════════════════════════════════════
def plot_evaluation(model_name, y_test, y_pred, y_proba):
    """
    Genere les graphiques d'evaluation :
    - Courbe ROC
    - Courbe Precision-Recall
    - Matrice de confusion
    """
    print("\n[ETAPE 3/6] Graphiques d evaluation...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Evaluation - {model_name}",
                 fontsize=14, fontweight="bold")

    # ── Courbe ROC ───────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    axes[0].plot(fpr, tpr, color="#2196F3", linewidth=2,
                 label=f"AUC = {auc:.3f}")
    axes[0].plot([0,1], [0,1], "k--", linewidth=1)
    axes[0].fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
    axes[0].set_xlabel("Taux Faux Positifs")
    axes[0].set_ylabel("Taux Vrais Positifs")
    axes[0].set_title("Courbe ROC")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ── Courbe Precision-Recall ───────────────
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    axes[1].plot(recall, precision, color="#F44336", linewidth=2,
                 label=f"AP = {ap:.3f}")
    axes[1].fill_between(recall, precision, alpha=0.1, color="#F44336")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Courbe Precision-Recall")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # ── Matrice de Confusion ─────────────────
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                ax=axes[2],
                xticklabels=["Survivant", "Decede"],
                yticklabels=["Survivant", "Decede"])
    axes[2].set_xlabel("Predit")
    axes[2].set_ylabel("Reel")
    axes[2].set_title("Matrice de Confusion")

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "evaluation_finale.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [plot] Evaluation finale sauvegardee")


# ══════════════════════════════════════════════
# 4. ANALYSE SHAP
# ══════════════════════════════════════════════
def compute_shap_values(model, X_test, feature_names, model_name):
    """
    Calcule les valeurs SHAP pour expliquer
    les predictions du modele.

    SHAP = SHapley Additive exPlanations
    Explique POURQUOI le modele a predit
    qu'un patient est a risque.
    """
    print("\n[ETAPE 4/6] Analyse SHAP...")

    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Choisir le bon explainer selon le modele
    model_type = model_name.lower()

    if "forest" in model_type or "xgb" in model_type \
            or "lgbm" in model_type or "lightgbm" in model_type \
            or "boost" in model_type:
        print(f"  Utilisation de TreeExplainer...")
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_df)

        # Pour Random Forest : shap_values peut etre une liste [classe0, classe1]
        # ou un array 3D de shape (n_samples, n_features, n_classes)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]  # classe 1 = Decede
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_vals = shap_values[:, :, 1]  # classe 1 = Decede
        else:
            shap_vals = shap_values

    else:
        print(f"  Utilisation de LinearExplainer...")
        explainer   = shap.LinearExplainer(model, X_test_df)
        shap_vals   = explainer.shap_values(X_test_df)

    print(f"  SHAP values calculees : {shap_vals.shape}")
    return explainer, shap_vals, X_test_df


def plot_shap_summary(shap_vals, X_test_df, model_name):
    """
    Graphique SHAP Summary Plot :
    Montre les features les plus importantes
    et leur impact sur la prediction.
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_vals,
        X_test_df,
        plot_type="bar",
        show=False,
        max_display=17
    )
    plt.title(f"SHAP - Importance des Features ({model_name})",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "shap_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] SHAP importance sauvegardee")


def plot_shap_beeswarm(shap_vals, X_test_df, model_name):
    """
    SHAP Beeswarm Plot :
    Montre l'impact de chaque feature
    sur chaque patient individuellement.
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_vals,
        X_test_df,
        show=False,
        max_display=17
    )
    plt.title(f"SHAP - Impact par Patient ({model_name})",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "shap_beeswarm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] SHAP beeswarm sauvegarde")


def plot_shap_dependence(shap_vals, X_test_df):
    """
    SHAP Dependence Plot :
    Montre la relation entre les 3 features
    les plus importantes et leur impact SHAP.
    """
    mean_shap = np.abs(shap_vals).mean(axis=0)
    top3_idx  = np.argsort(mean_shap)[::-1][:3]
    top3_feat = [str(X_test_df.columns[i]) for i in top3_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("SHAP - Top 3 Features",
                 fontsize=13, fontweight="bold")

    for i, feat in enumerate(top3_feat):
        feat = str(feat)
        feat_idx = list(X_test_df.columns).index(feat)
        axes[i].scatter(
            X_test_df[feat],
            shap_vals[:, feat_idx],
            alpha=0.6, color="#2196F3"
        )
        axes[i].axhline(y=0, color="red", linestyle="--")
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel("Valeur SHAP")
        axes[i].set_title(f"{feat}")
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "shap_dependence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] SHAP dependence sauvegarde")


def plot_shap_waterfall(explainer, shap_vals, X_test_df, model_name):
    """
    SHAP Waterfall Plot :
    Explique la prediction pour UN patient
    specifique (le plus a risque).
    """
    # Patient le plus a risque
    risk_idx = int(np.argmax(shap_vals.sum(axis=1)))

    plt.figure(figsize=(10, 7))

    # Trier les features par importance pour ce patient
    patient_shap = shap_vals[risk_idx]
    sorted_idx   = np.argsort(np.abs(patient_shap))[::-1][:10]

    colors = ["#F44336" if v > 0 else "#2196F3"
              for v in patient_shap[sorted_idx]]

    plt.barh(
        [X_test_df.columns[i] for i in sorted_idx[::-1]],
        patient_shap[sorted_idx[::-1]],
        color=colors[::-1]
    )
    plt.axvline(x=0, color="black", linewidth=0.8)
    plt.xlabel("Valeur SHAP (impact sur la prediction)")
    plt.title(f"SHAP - Explication Patient a Risque\n({model_name})",
              fontsize=13, fontweight="bold")
    plt.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "shap_waterfall.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] SHAP waterfall sauvegarde")


# ══════════════════════════════════════════════
# 5. RAPPORT FINAL
# ══════════════════════════════════════════════
def generate_report(model_name, metrics, shap_vals, feature_names):
    """
    Genere un rapport texte avec :
    - Les metriques finales
    - Le top 5 des features SHAP
    """
    print("\n[ETAPE 5/6] Generation du rapport...")

    # Top 5 features SHAP
    mean_shap = np.abs(shap_vals).mean(axis=0)
    sorted_idx = np.argsort(mean_shap)[::-1]
    top5 = [(feature_names[i], round(float(mean_shap[i]), 4))
            for i in sorted_idx[:5]]

    report = f"""
╔══════════════════════════════════════════╗
║     RAPPORT FINAL - HEART FAILURE        ║
╠══════════════════════════════════════════╣
║  Meilleur Modele : {model_name:<23}║
╠══════════════════════════════════════════╣
║  METRIQUES FINALES                       ║
╠══════════════════════════════════════════╣
║  ROC-AUC           : {metrics['ROC-AUC']:<21}║
║  Accuracy          : {metrics['Accuracy']:<21}║
║  Precision         : {metrics['Precision']:<21}║
║  Recall            : {metrics['Recall']:<21}║
║  F1-Score          : {metrics['F1-Score']:<21}║
╠══════════════════════════════════════════╣
║  TOP 5 FEATURES SHAP                     ║
╠══════════════════════════════════════════╣"""

    for i, (feat, val) in enumerate(top5, 1):
        report += f"\n║  {i}. {feat:<20} : {val:<16}║"

    report += """
╠══════════════════════════════════════════╣
║  GRAPHIQUES GENERES                      ║
╠══════════════════════════════════════════╣
║  -> evaluation_finale.png                ║
║  -> shap_importance.png                  ║
║  -> shap_beeswarm.png                    ║
║  -> shap_dependence.png                  ║
║  -> shap_waterfall.png                   ║
╚══════════════════════════════════════════╝
"""
    print(report)

    # Sauvegarder le rapport
    path = os.path.join(MODEL_DIR, "rapport_final.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  [save] Rapport sauvegarde -> rapport_final.txt")

    return top5


# ══════════════════════════════════════════════
# 6. PIPELINE COMPLET D'EVALUATION
# ══════════════════════════════════════════════
def run_evaluation_pipeline():
    """
    Pipeline complet d'evaluation :
    1. Charger le meilleur modele
    2. Evaluer sur le test set
    3. Graphiques d'evaluation
    4. Analyse SHAP
    5. Rapport final
    """
    print("\n" + "="*55)
    print("  HEART FAILURE - EVALUATION PIPELINE")
    print("="*55)

    # ETAPE 1 : Charger le modele
    model, scaler, feature_names, model_name = load_best_model()

    # ETAPE 2 : Preparer la data
    print("\n[ETAPE 2/6] Preparation du test set...")
    result  = run_full_pipeline(DATA_PATH)
    X_test  = result["X_test_scaled"]
    y_test  = result["y_test"]
    print(f"  X_test : {X_test.shape}")
    print(f"  y_test : {dict(y_test.value_counts())}")

    # ETAPE 3 : Evaluer
    metrics, y_pred, y_proba = evaluate_best_model(
        model, X_test, y_test, feature_names)

    # ETAPE 4 : Graphiques evaluation
    plot_evaluation(model_name, y_test, y_pred, y_proba)

    # ETAPE 5 : SHAP
    print("\n[ETAPE 5/6] Analyse SHAP...")
    explainer, shap_vals, X_test_df = compute_shap_values(
        model, X_test, feature_names, model_name)

    plot_shap_summary(shap_vals, X_test_df, model_name)
    plot_shap_beeswarm(shap_vals, X_test_df, model_name)
    plot_shap_dependence(shap_vals, X_test_df)
    plot_shap_waterfall(explainer, shap_vals, X_test_df, model_name)

    # ETAPE 6 : Rapport
    top5 = generate_report(model_name, metrics, shap_vals, feature_names)

    # RESUME FINAL
    print("\n" + "="*55)
    print("  EVALUATION PIPELINE COMPLET !")
    print("="*55)
    print(f"\n  Meilleur modele : {model_name}")
    print(f"\n  Graphiques dans notebooks/figures/ :")
    print(f"    -> evaluation_finale.png")
    print(f"    -> shap_importance.png")
    print(f"    -> shap_beeswarm.png")
    print(f"    -> shap_dependence.png")
    print(f"    -> shap_waterfall.png")
    print(f"\n  Rapport dans models/ :")
    print(f"    -> rapport_final.txt")
    print("="*55)

    return model, metrics, shap_vals, top5


# ══════════════════════════════════════════════
# LANCER
# ══════════════════════════════════════════════
if __name__ == "__main__":
    model, metrics, shap_vals, top5 = run_evaluation_pipeline()