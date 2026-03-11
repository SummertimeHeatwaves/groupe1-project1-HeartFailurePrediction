import streamlit as st

# Configuration de la page
st.set_page_config(page_title="Heart Failure Risk Predictor", layout="wide")

st.title("Heart Failure Risk Predictor")

st.markdown("### Saisie des données cliniques du patient")

# Création de 3 colonnes pour le layout
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

# Bouton de prédiction
if st.button("Predict Heart Failure Risk", type="primary", use_container_width=True):
    st.success("Le formulaire fonctionne ! (La logique de prédiction sera ajoutée plus tard)")