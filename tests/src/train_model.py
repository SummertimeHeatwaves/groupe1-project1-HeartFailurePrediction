import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import logging

# Permet d'importer depuis la racine du projet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_processing import run_preprocessing_pipeline

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def get_models() -> dict:
    """Retourne un dictionnaire des modèles de ML pré-configurés."""
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=2.1,
            random_state=42, eval_metric="logloss"
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1, num_leaves=31,
            min_child_samples=10, class_weight="balanced", random_state=42, verbose=-1
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=42
        )
    }
    return models

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
    """Évalue un modèle entraîné sur les données de test."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    
    logger.info(f"\n{'='*55}")
    logger.info(f" Model: {model_name}")
    logger.info(f" Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f" Precision: {metrics['precision']:.4f}")
    logger.info(f" Recall: {metrics['recall']:.4f}")
    logger.info(f" F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f" ROC-AUC: {metrics['roc_auc']:.4f}")
    return metrics

def cross_validate_model(model, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> float:
    """Effectue une validation croisée k-fold stratifiée."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")
    mean_auc = round(scores.mean(), 4)
    std_auc = round(scores.std(), 4)
    logger.info(f" CV ROC-AUC: {mean_auc} (+/- {std_auc})")
    return mean_auc

def train_all_models(data: dict) -> tuple:
    """Entraîne tous les modèles et sélectionne le meilleur."""
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    models = get_models()
    all_results = []
    trained_models = {}

    for name, model in models.items():
        logger.info(f"\n{'='*55}")
        logger.info(f"--- Training: {name} ---")
        
        # Entraînement sur tout le set
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Score de cross-validation
        cv_auc = cross_validate_model(model, X_train, y_train)
        
        # Évaluation sur le set de test
        metrics = evaluate_model(model, X_test, y_test, name)
        metrics["cv_roc_auc"] = cv_auc
        all_results.append(metrics)

    # Sélectionner le meilleur selon ROC-AUC
    best_result = max(all_results, key=lambda x: x["roc_auc"])
    best_name = best_result["model_name"]
    
    logger.info(f"\n{'='*55}")
    logger.info(f" BEST MODEL: {best_name}")
    logger.info(f" ROC-AUC: {best_result['roc_auc']}")
    logger.info(f" {'='*55}")
    
    return all_results, trained_models, best_name

def save_artifacts(model, scaler, feature_names: list, results: list, model_dir: str = "models"):
    """Sauvegarde les modèles et scalers pour le déploiement."""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(model_dir, "feature_names.pkl"))
    
    with open(os.path.join(model_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"All artifacts saved to {model_dir}/")

if __name__ == "__main__":
    # Étape 1: Pré-traitement
    logger.info("Step 1: Running preprocessing pipeline...")
    data = run_preprocessing_pipeline()
    
    # Étape 2: Entraînement
    logger.info("Step 2: Training all models...")
    results, trained_models, best_name = train_all_models(data)
    
    # Étape 3: Sauvegarde
    logger.info("Step 3: Saving artifacts...")
    save_artifacts(
        model=trained_models[best_name],
        scaler=data["scaler"],
        feature_names=data["feature_names"],
        results=results
    )
    
    # Étape 4: Tableau de comparaison
    print("\n" + "=" * 72)
    print(f" {'Model':<22} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8} {'CV-AUC':>8} ")
    print("-" * 72)
    for r in results:
        mark = "***" if r["model_name"] == best_name else ""
        print(
            f" {r['model_name']:<22}"
            f" {r['accuracy']:>8.4f} {r['precision']:>8.4f} "
            f" {r['recall']:>8.4f} {r['f1_score']:>8.4f} "
            f" {r['roc_auc']:>8.4f} {r['cv_roc_auc']:>8.4f} {mark}"
        )
    print(f"\nBest: {best_name}")