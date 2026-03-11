import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(page_title="Heart Failure Risk Predictor", layout="wide")

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_model_artifacts():
    """Charge le modèle, le scaler et les noms des features"""
    # Remonte d'un dossier depuis 'app' pour trouver 'models'
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    features = joblib.load(os.path.join(model_dir, "feature_names.pkl"))
    return model, scaler, features

try:
    model, scaler, feature_names = load_model_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Impossible de charger le modèle : {e}. Assurez-vous que le dossier 'models' existe.")

# --- INTERFACE UTILISATEUR ---
st.title("Heart Failure Risk Predictor")
st.markdown("### Saisie des données cliniques du patient")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Demographics")
    age = st.number_input("Age (years)", min_value=18, max_value=120, value=60)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

with col2:
    st.markdown("#### Clinical Markers")
    ejection_fraction = st.slider("Ejection Fraction (%)", 10, 80, 38)
    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.1, 10.0, 1.1, 0.1)
    serum_sodium = st.number_input("Serum Sodium (mEq/L)", 100, 150, 137)
    creatinine_phosphokinase = st.number_input("CPK Enzyme (mcg/L)", 20, 8000, 250)

with col3:
    st.markdown("#### Additional")
    anaemia = st.selectbox("Anaemia", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    platelets = st.number_input("Platelets (kiloplatelets/mL)", 25000.0, 900000.0, 263358.0, 1000.0)

st.markdown("---")

# --- LOGIQUE DE PRÉDICTION ---
if st.button("Predict Heart Failure Risk", type="primary", use_container_width=True):
    if not model_loaded:
        st.warning("Le modèle n'est pas chargé.")
    else:
        # 1. Préparer les données saisies
        patient_data = {
            "age": age, "anaemia": anaemia,
            "creatinine_phosphokinase": creatinine_phosphokinase,
            "diabetes": diabetes, "ejection_fraction": ejection_fraction,
            "high_blood_pressure": high_blood_pressure, "platelets": platelets,
            "serum_creatinine": serum_creatinine, "serum_sodium": serum_sodium,
            "sex": sex, "smoking": smoking
        }
        
        # 2. Mise à l'échelle (Scaling)
        input_df = pd.DataFrame([patient_data])[feature_names]
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
        
        # 3. Prédiction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # 4. Affichage du Niveau de Risque
        st.markdown("### Résultats de la Prédiction")
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if probability >= 0.7:
                st.error("🚨 **HAUT RISQUE** détecté")
            elif probability >= 0.4:
                st.warning("⚠️ **RISQUE MODÉRÉ** détecté")
            else:
                st.success("✅ **FAIBLE RISQUE** détecté")
        
        with col_res2:
            st.metric("Probabilité d'insuffisance cardiaque", f"{probability * 100:.1f} %")
            
        # 5. Explicabilité avec SHAP
        st.markdown("---")
        st.markdown("#### Explication détaillée (Impact des variables)")
        with st.spinner("Analyse en cours..."):
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer(input_scaled_df)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_vals[0], show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()