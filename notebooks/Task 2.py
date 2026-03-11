"""
TÂCHE 2 : Apply SMOTE to fix the class imbalance
À coller dans une cellule de notebooks/eda.ipynb
"""

# ================================================================
# ÉTAPE 1 : Imports
# ================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# ================================================================
# ÉTAPE 2 : Charger le dataset
# ================================================================
df = pd.read_csv("data/heart_failure_clinical_records.csv")

print("Shape du dataset:", df.shape)
print("\nDistribution AVANT SMOTE:")
print(df["DEATH_EVENT"].value_counts())
print(df["DEATH_EVENT"].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")

# ================================================================
# ÉTAPE 3 : Visualiser le déséquilibre AVANT
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Graphique AVANT
counts_before = df["DEATH_EVENT"].value_counts()
axes[0].bar(
    ["Survived (0)", "Deceased (1)"],
    counts_before.values,
    color=["#2ECC71", "#E74C3C"],
    edgecolor="white",
    linewidth=1.5
)
axes[0].set_title("AVANT SMOTE\nDistribution des classes", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Nombre de patients")
for i, v in enumerate(counts_before.values):
    axes[0].text(i, v + 2, f"{v}\n({v/len(df)*100:.1f}%)",
                 ha="center", fontweight="bold")
axes[0].set_ylim(0, 250)

# ================================================================
# ÉTAPE 4 : Préparer les features et split train/test
# ================================================================
feature_cols = [c for c in df.columns if c not in ["DEATH_EVENT", "time"]]
X = df[feature_cols]
y = df["DEATH_EVENT"]

# Split AVANT SMOTE (SMOTE s'applique uniquement sur le train !)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y        # Garder les proportions dans chaque split
)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
print(f"\nDistribution y_train AVANT SMOTE:\n{y_train.value_counts()}")

# ================================================================
# ÉTAPE 5 : Appliquer SMOTE sur le TRAIN uniquement
# ================================================================
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"\nDistribution y_train APRÈS SMOTE:\n{y_train_resampled.value_counts()}")
print(f"\nTaille train AVANT SMOTE : {len(X_train)}")
print(f"Taille train APRÈS SMOTE : {len(X_train_resampled)}")
print(f"Nouveaux échantillons synthétiques créés : {len(X_train_resampled) - len(X_train)}")

# ================================================================
# ÉTAPE 6 : Visualiser APRÈS SMOTE
# ================================================================
counts_after = pd.Series(y_train_resampled).value_counts()
axes[1].bar(
    ["Survived (0)", "Deceased (1)"],
    counts_after.values,
    color=["#2ECC71", "#E74C3C"],
    edgecolor="white",
    linewidth=1.5
)
axes[1].set_title("APRÈS SMOTE\nDistribution du train set", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Nombre de patients")
for i, v in enumerate(counts_after.values):
    axes[1].text(i, v + 2, f"{v}\n({v/len(y_train_resampled)*100:.1f}%)",
                 ha="center", fontweight="bold")
axes[1].set_ylim(0, 250)

plt.suptitle("Effet du SMOTE sur l'équilibre des classes", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("reports/figures/smote_balance.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✅ SMOTE appliqué avec succès !")
print(f"   Le test set reste intact : {pd.Series(y_test).value_counts().to_dict()}")
print("   → Pas de data leakage : SMOTE appliqué uniquement sur le train.")
