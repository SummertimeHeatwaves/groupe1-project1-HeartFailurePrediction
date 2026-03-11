# ══════════════════════════════════════════════════════════════
#   SHAP SUMMARY PLOTS — WINNING MODEL
#   Compatible : XGBoost / Random Forest / LightGBM / LogReg
# ══════════════════════════════════════════════════════════════

import shap
import matplotlib.pyplot as plt
import numpy as np

# ── 1. Créer l'explainer selon le type de modèle ─────────────
def get_shap_explainer(model, X_train):
    model_name = type(model).__name__

    if model_name in ['XGBClassifier', 'LGBMClassifier']:
        explainer = shap.TreeExplainer(model)

    elif model_name == 'RandomForestClassifier':
        explainer = shap.TreeExplainer(model)

    elif model_name == 'LogisticRegression':
        explainer = shap.LinearExplainer(model, X_train)

    else:
        # Fallback universel (plus lent)
        explainer = shap.KernelExplainer(model.predict_proba, 
                                          shap.sample(X_train, 100))
    return explainer


# ── 2. Calculer les SHAP values ───────────────────────────────
explainer  = get_shap_explainer(best_model, X_train)
shap_values = explainer.shap_values(X_test)

# Pour les modèles binaires (RandomForest retourne 2 classes)
if isinstance(shap_values, list):
    shap_vals = shap_values[1]   # classe 1 = Décédé
else:
    shap_vals = shap_values

feature_names = list(X.columns)   # ← remplace X par ton DataFrame features


# ══════════════════════════════════════════════════════════════
#   PLOT 1 — Summary Plot (Beeswarm)
#   → Montre l'impact global de chaque feature
# ══════════════════════════════════════════════════════════════
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_vals,
    X_test,
    feature_names=feature_names,
    plot_type='dot',          # beeswarm
    show=False,
    max_display=12
)
plt.title('🫀 SHAP Summary Plot — Impact des Features sur la Prédiction',
          fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('shap_summary_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ Sauvegardé : shap_summary_beeswarm.png')


# ══════════════════════════════════════════════════════════════
#   PLOT 2 — Bar Plot (Importance globale)
#   → Classement des features par importance moyenne |SHAP|
# ══════════════════════════════════════════════════════════════
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_vals,
    X_test,
    feature_names=feature_names,
    plot_type='bar',
    show=False,
    max_display=12
)
plt.title('📊 SHAP Feature Importance (Mean |SHAP value|)',
          fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('shap_summary_bar.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ Sauvegardé : shap_summary_bar.png')


# ══════════════════════════════════════════════════════════════
#   PLOT 3 — Waterfall Plot (1 patient individuel)
#   → Explication d'UNE prédiction spécifique
# ══════════════════════════════════════════════════════════════
patient_idx = 0    # ← change l'index pour voir un autre patient

shap_explanation = shap.Explanation(
    values       = shap_vals[patient_idx],
    base_values  = explainer.expected_value if not isinstance(
                       explainer.expected_value, list)
                   else explainer.expected_value[1],
    data         = X_test.iloc[patient_idx] if hasattr(X_test, 'iloc')
                   else X_test[patient_idx],
    feature_names= feature_names
)

plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_explanation, show=False, max_display=12)
plt.title(f'🔍 SHAP Waterfall — Patient #{patient_idx}',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'shap_waterfall_patient{patient_idx}.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Sauvegardé : shap_waterfall_patient{patient_idx}.png')


# ══════════════════════════════════════════════════════════════
#   PLOT 4 — Force Plot (Top 3 features les plus importantes)
#   → Visualisation des features qui poussent vers 0 ou 1
# ══════════════════════════════════════════════════════════════
mean_shap = np.abs(shap_vals).mean(axis=0)
top3_idx  = np.argsort(mean_shap)[::-1][:3]
top3_names = [feature_names[i] for i in top3_idx]

print('\n📌 Top 3 features les plus importantes (SHAP) :')
for i, (idx, name) in enumerate(zip(top3_idx, top3_names)):
    print(f'  {i+1}. {name:<30} → mean|SHAP| = {mean_shap[idx]:.4f}')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ['#e74c3c', '#e67e22', '#3498db']

for ax, feat_idx, feat_name, color in zip(axes, top3_idx, top3_names, colors):
    feat_vals  = X_test.iloc[:, feat_idx] if hasattr(X_test, 'iloc') \
                 else X_test[:, feat_idx]
    shap_col   = shap_vals[:, feat_idx]

    sc = ax.scatter(feat_vals, shap_col,
                    c=feat_vals, cmap='RdYlGn_r',
                    alpha=0.6, edgecolors='none', s=40)
    ax.axhline(y=0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel(feat_name, fontsize=11, fontweight='bold')
    ax.set_ylabel('SHAP value', fontsize=10)
    ax.set_title(f'Dependence: {feat_name}',
                 fontweight='bold', fontsize=11, color=color)
    plt.colorbar(sc, ax=ax, label=feat_name)

fig.suptitle('🔗 SHAP Dependence Plots — Top 3 Features',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('shap_dependence_top3.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ Sauvegardé : shap_dependence_top3.png')