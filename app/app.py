import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioPredict AI | Coding Week 2026",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design CSS Adaptatif ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .hero-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        color: white !important;
    }
    .result-card-danger {
        background-color: #fef2f2;
        border: 2px solid #f87171;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        color: #991b1b;
    }
    .result-card-safe {
        background-color: #f0fdf4;
        border: 2px solid #4ade80;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        color: #166534;
    }
    .result-value {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Fonction utilitaire pour trouver les images ───────────────────────────────
def get_image_path(filename):
    """Cherche l'image dans tous les dossiers possibles du projet."""
    paths = [
        f"reports/figures/{filename}",
        f"../reports/figures/{filename}",
        f"notebooks/figures/{filename}",
        f"../notebooks/figures/{filename}",
        filename
    ]
    for path in paths:
        if os.path.exists(path):
            return path
    return None

# ── Chargement des Modèles ────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("models/best_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        feature_names = joblib.load("models/feature_names.pkl")
        return model, scaler, feature_names
    except Exception as e:
        st.error("⚠️ Modèle introuvable. Exécutez 'python src/train_model.py' en premier.")
        return None, None, None

assets = load_assets()
if assets[0] is not None:
    model, scaler, feature_names = assets

def build_patient(age, anaemia, creatinine_phosphokinase, diabetes,
                  ejection_fraction, high_blood_pressure, platelets,
                  serum_creatinine, serum_sodium, sex, smoking, time):
    base = {
        'age': age, 'anaemia': anaemia, 'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes': diabetes, 'ejection_fraction': ejection_fraction, 'high_blood_pressure': high_blood_pressure,
        'platelets': platelets, 'serum_creatinine': serum_creatinine, 'serum_sodium': serum_sodium,
        'sex': sex, 'smoking': smoking, 'time': time,
        'age_group': 0 if age < 50 else (1 if age < 65 else 2),
        'ef_severity': 0 if ejection_fraction >= 50 else (1 if ejection_fraction >= 40 else 2),
        'sodium_risk': 1 if serum_sodium < 135 else 0,
        'cp_log': np.log1p(creatinine_phosphokinase),
        'platelet_log': np.log1p(platelets),
    }
    return pd.DataFrame([base])[feature_names]

# ── Menu Latéral ──────────────────────────────────────────────────────────────
st.sidebar.title("🫀 CardioPredict")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation :",
    ["🔮 1. Prédiction Patient", "📊 2. Dataset Info", "📈 3. Performance Modèle"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Projet Coding Week 2026**\n\n"
    "École Centrale Casablanca\n\n"
    "Développé par : Groupe 1"
)

# ==============================================================================
# PAGE 1 : PRÉDICTION
# ==============================================================================
if page == "🔮 1. Prédiction Patient" and assets[0] is not None:
    
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">CardioPredict AI</h1>
        <p style="margin-top:10px; font-size:1.1rem;">Système de support à la décision clinique propulsé par Machine Learning et SHAP.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📝 Saisie des paramètres cliniques")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Démographie & Habitudes**")
        age = st.slider("Âge", 40, 95, 60)
        sex = st.selectbox("Sexe", [1, 0], format_func=lambda x: "Homme" if x else "Femme")
        smoking = st.selectbox("Tabagisme", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
        time = st.slider("Période de suivi (jours)", 4, 285, 100)

    with col2:
        st.markdown("**Antécédents**")
        diabetes = st.selectbox("Diabète", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
        high_blood_pressure = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
        anaemia = st.selectbox("Anémie", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")

    with col3:
        st.markdown("**Biologie**")
        ejection_fraction = st.slider("Fraction d'éjection (%)", 14, 80, 38)
        serum_creatinine = st.number_input("Créatinine sérique (mg/dL)", 0.5, 9.4, 1.1, step=0.1)
        serum_sodium = st.slider("Sodium sérique (mEq/L)", 113, 148, 137)
        creatinine_phosphokinase = st.number_input("CPK", 23, 7861, 250)
        platelets = st.number_input("Plaquettes", 25000, 850000, 250000, step=1000)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 LANCER L'ANALYSE PRÉDICTIVE", use_container_width=True, type="primary")

    if predict_btn:
        patient_data = build_patient(
            age, anaemia, creatinine_phosphokinase, diabetes,
            ejection_fraction, high_blood_pressure, platelets,
            serum_creatinine, serum_sodium, sex, smoking, time
        )

        X_scaled = scaler.transform(patient_data)
        input_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
        
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]

        st.markdown("---")
        st.subheader("Résultat du Diagnostic IA")

        res_col1, res_col2 = st.columns([1, 1.5])
        
        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-card-danger">
                    <h3>⚠️ RISQUE ÉLEVÉ</h3>
                    <div class="result-value">{probability[1]*100:.1f}%</div>
                    <p>Mortalité prédite. Prise en charge recommandée.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-safe">
                    <h3>✅ RISQUE FAIBLE</h3>
                    <div class="result-value">{probability[0]*100:.1f}%</div>
                    <p>Survie prédite. Profil clinique stable.</p>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("### ⚡ Explicabilité IA (SHAP)")
            st.caption("Le graphique Waterfall montre comment chaque paramètre pousse la prédiction de la valeur moyenne (Base Value) vers la probabilité finale.")
        with st.spinner("Génération du graphique SHAP..."):
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer(input_scaled_df)
                
                # Sélection robuste de la classe 1 (Décès) pour le graphique
                exp_obj = shap_vals[0]
                if len(exp_obj.values.shape) > 1:
                    exp_obj = shap_vals[0, :, 1]
                    
                fig_wf = plt.figure(figsize=(8, 5))
                shap.plots.waterfall(exp_obj, show=False)
                plt.tight_layout()
                st.pyplot(fig_wf)
                plt.close()

        st.markdown("#### Détails des Contributions")
        shap_array = explainer.shap_values(X_scaled)
        if isinstance(shap_array, list):
            sv = shap_array[1][0]
        elif len(np.array(shap_array).shape) == 3:
            sv = shap_array[0, :, 1]
        else:
            sv = shap_array[0]
        sv = np.ravel(sv)
        
        contributions = []
        for i, feat in enumerate(feature_names):
            val_shap = sv[i]
            contributions.append({
                "Paramètre": feat,
                "Valeur (Standardisée)": round(X_scaled[0][i], 2),
                "Impact SHAP": round(val_shap, 4),
                "Effet": "⬆️ Augmente le risque" if val_shap > 0 else "⬇️ Diminue le risque",
                "Abs_Impact": abs(val_shap)
            })
            
        contrib_df = pd.DataFrame(contributions).sort_values("Abs_Impact", ascending=False).drop(columns=["Abs_Impact"])
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 2 : DATASET INFO
# ==============================================================================
elif page == "📊 2. Dataset Info":
    st.subheader("📊 Informations sur le Dataset")
    st.info("""
    **Heart Failure Clinical Records Dataset (UCI)**
    - **Taille :** 299 patients, 13 variables
    - **Déséquilibre initial :** 68% Survie (Class 0), 32% Décès (Class 1)
    - **Traitement :** Oversampling **SMOTE** appliqué sur le jeu d'entraînement pour atteindre un équilibre 50/50.
    """)
    
    features_info = pd.DataFrame({
        "Feature": ["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking", "time"],
        "Type": ["Continu", "Binaire", "Continu", "Binaire", "Continu", "Binaire", "Continu", "Continu", "Continu", "Binaire", "Binaire", "Continu"],
        "Description": ["Âge du patient", "Baisse des globules rouges", "Niveau de CPK", "Patient diabétique", "Pourcentage de sang quittant le cœur", "Hypertension", "Plaquettes", "Créatinine dans le sang", "Sodium dans le sang", "0=Femme, 1=Homme", "Fumeur", "Période de suivi (exclue du modèle)"]
    })
    st.dataframe(features_info, use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 3 : MODEL PERFORMANCE
# ==============================================================================
elif page == "📈 3. Performance Modèle":
    st.subheader("📈 Comparaison des Modèles")
    
    st.markdown("Le **Random Forest** a été sélectionné comme modèle final car il maximise le ROC-AUC tout en conservant un excellent Recall (critère vital en médecine).")
    
    try:
        df_res = pd.DataFrame()
        if os.path.exists("models/metrics.csv"):
            df_res = pd.read_csv("models/metrics.csv")
        elif os.path.exists("models/results.json"):
            with open("models/results.json", "r") as f:
                df_res = pd.DataFrame(json.load(f))
                
        if not df_res.empty:
            model_col = next((col for col in df_res.columns if col.lower() in ["model_name", "model", "modèle", "nom"]), None)
            df_display = df_res.set_index(model_col) if model_col else df_res
            st.dataframe(df_display, use_container_width=True)
        else:
            st.warning("Aucun fichier de métriques trouvé dans le dossier 'models/'.")
    except Exception as e:
        st.error(f"Erreur d'affichage : {e}")

    st.markdown("---")
    
    # --- AJOUT DES NOUVEAUX GRAPHIQUES ICI ---
    
    # Première ligne de graphiques : ROC & Matrices de confusion
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### Courbes ROC")
        roc_img = get_image_path("roc_curves.png") or get_image_path("roc_curve.png")
        if roc_img:
            st.image(roc_img, use_container_width=True)
        else:
            st.info("⚠️ Graphique ROC introuvable.")
            
    with c2:
        st.markdown("### Matrices de Confusion")
        conf_img = get_image_path("confusion_matrices.png")
        if conf_img:
            st.image(conf_img, use_container_width=True)
        else:
            st.info("⚠️ Matrices de confusion introuvables.")

    st.markdown("---")
    
    # Deuxième ligne de graphiques : Métriques & SHAP
    c3, c4 = st.columns(2)
    
    with c3:
        st.markdown("### Comparaison des Métriques")
        met_img = get_image_path("metrics_comparison.png")
        if met_img:
            st.image(met_img, use_container_width=True)
        else:
            st.info("⚠️ Comparaison des métriques introuvable.")

    with c4:
        st.markdown("### Importance Globale (SHAP Summary)")
        shap_img = get_image_path("shap_summary.png") or get_image_path("shap_summary_beeswarm.png")
        if shap_img:
            st.image(shap_img, use_container_width=True)
        else:
            st.info("⚠️ Graphique SHAP introuvable.")