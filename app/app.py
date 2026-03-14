import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioPredict AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #0a0a0f; color: #e8e6f0; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12121a 0%, #0d0d15 100%);
        border-right: 1px solid rgba(220, 38, 38, 0.2);
    }
    [data-testid="stSidebar"] label {
        color: #a0aec0 !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }

    .hero-header {
        background: linear-gradient(135deg, #1a0a0a 0%, #0f0a1a 50%, #0a0f1a 100%);
        border: 1px solid rgba(220, 38, 38, 0.15);
        border-radius: 20px;
        padding: 48px 56px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%; right: -10%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(220,38,38,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 3.2rem;
        color: #ffffff;
        margin: 0 0 8px 0;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .hero-title span { color: #dc2626; }
    .hero-subtitle { font-size: 1rem; color: #718096; font-weight: 300; margin: 0; }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(220,38,38,0.1);
        border: 1px solid rgba(220,38,38,0.3);
        border-radius: 100px;
        padding: 4px 14px;
        font-size: 0.75rem;
        color: #fc8181;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 20px;
    }
    .hero-badge::before { content: '●'; font-size: 0.5rem; animation: pulse 2s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 32px;
    }
    .metric-card {
        background: linear-gradient(135deg, #13131d 0%, #0f0f18 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 24px;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #dc2626, transparent);
    }
    .metric-label { font-size: 0.7rem; color: #4a5568; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px; }
    .metric-value { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #ffffff; line-height: 1; margin-bottom: 4px; }
    .metric-sub { font-size: 0.75rem; color: #4a5568; }

    .section-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.5rem;
        color: #e8e6f0;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(220,38,38,0.3), transparent);
        margin-left: 10px;
    }

    .patient-card {
        background: #13131d;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 28px;
    }

    .prediction-high {
        background: linear-gradient(135deg, rgba(220,38,38,0.15), rgba(153,27,27,0.1));
        border: 1px solid rgba(220,38,38,0.4);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-low {
        background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(6,95,70,0.1));
        border: 1px solid rgba(16,185,129,0.4);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin: 20px 0;
    }

    .graph-card {
        background: #13131d;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .graph-title { font-size: 0.75rem; color: #4a5568; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 16px; }

    .stButton > button {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        width: 100% !important;
        box-shadow: 0 4px 20px rgba(220,38,38,0.3) !important;
    }
    .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 30px rgba(220,38,38,0.4) !important; }

    .stTabs [data-baseweb="tab-list"] {
        background: #13131d !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 4px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
    }
    .stTabs [data-baseweb="tab"] { background: transparent !important; color: #718096 !important; border-radius: 8px !important; font-weight: 500 !important; }
    .stTabs [aria-selected="true"] { background: rgba(220,38,38,0.15) !important; color: #fc8181 !important; }

    .feature-bar { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
    .feature-name { font-size: 0.8rem; color: #a0aec0; width: 180px; flex-shrink: 0; }
    .feature-track { flex: 1; height: 6px; background: rgba(255,255,255,0.05); border-radius: 10px; overflow: hidden; }
    .feature-fill { height: 100%; border-radius: 10px; }
    .feature-score { font-size: 0.75rem; color: #718096; width: 45px; text-align: right; }

    hr { border-color: rgba(255,255,255,0.05) !important; margin: 24px 0 !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0a0f; }
    ::-webkit-scrollbar-thumb { background: #2d2d3a; border-radius: 3px; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_assets()

# ── Build patient data with engineered features ───────────────────────────────
def build_patient(age, anaemia, creatinine_phosphokinase, diabetes,
                  ejection_fraction, high_blood_pressure, platelets,
                  serum_creatinine, serum_sodium, sex, smoking, time):
    base = {
        'age': age,
        'anaemia': anaemia,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes': diabetes,
        'ejection_fraction': ejection_fraction,
        'high_blood_pressure': high_blood_pressure,
        'platelets': platelets,
        'serum_creatinine': serum_creatinine,
        'serum_sodium': serum_sodium,
        'sex': sex,
        'smoking': smoking,
        'time': time,
        # ── Features calculées ──
        'age_group':    0 if age < 50 else (1 if age < 65 else 2),
        'ef_severity':  0 if ejection_fraction >= 50 else (1 if ejection_fraction >= 40 else 2),
        'sodium_risk':  1 if serum_sodium < 135 else 0,
        'cp_log':       np.log1p(creatinine_phosphokinase),
        'platelet_log': np.log1p(platelets),
    }
    return pd.DataFrame([base])[feature_names]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:\'DM Serif Display\',serif;font-size:1.3rem;color:#fff;margin-bottom:4px;">🫀 CardioPredict</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.75rem;color:#4a5568;margin-bottom:24px;letter-spacing:0.05em;">CLINICAL DECISION SUPPORT · AI</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Données du Patient**")

    age = st.slider("Âge", 40, 95, 60)
    sex = st.selectbox("Sexe", [1, 0], format_func=lambda x: "👨 Homme" if x else "👩 Femme")
    anaemia = st.selectbox("Anémie", [0, 1], format_func=lambda x: "✅ Non" if x == 0 else "⚠️ Oui")
    diabetes = st.selectbox("Diabète", [0, 1], format_func=lambda x: "✅ Non" if x == 0 else "⚠️ Oui")
    high_blood_pressure = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "✅ Non" if x == 0 else "⚠️ Oui")
    smoking = st.selectbox("Tabagisme", [0, 1], format_func=lambda x: "✅ Non" if x == 0 else "⚠️ Oui")

    st.markdown("---")
    st.markdown("**Paramètres Biologiques**")

    ejection_fraction = st.slider("Fraction d'éjection (%)", 14, 80, 38)
    serum_creatinine = st.number_input("Créatinine sérique (mg/dL)", 0.5, 9.4, 1.1, step=0.1)
    serum_sodium = st.slider("Sodium sérique (mEq/L)", 113, 148, 137)
    creatinine_phosphokinase = st.number_input("Créatinine Phosphokinase", 23, 7861, 250)
    platelets = st.number_input("Plaquettes (kiloplatelets/mL)", 25000, 850000, 250000, step=1000)
    time = st.slider("Période de suivi (jours)", 4, 285, 100)

    st.markdown("---")
    predict_btn = st.button("🔮 Analyser le Patient", use_container_width=True)

# ── Build patient ─────────────────────────────────────────────────────────────
patient_data = build_patient(
    age, anaemia, creatinine_phosphokinase, diabetes,
    ejection_fraction, high_blood_pressure, platelets,
    serum_creatinine, serum_sodium, sex, smoking, time
)

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">IA Médicale · Random Forest · SHAP Explainability</div>
    <h1 class="hero-title">Cardio<span>Predict</span> AI</h1>
    <p class="hero-subtitle">Outil avancé d'aide à la décision clinique pour la prédiction du risque d'insuffisance cardiaque</p>
</div>
""", unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-label">Meilleur Modèle</div>
        <div class="metric-value" style="font-size:1.3rem;margin-top:4px;">Random Forest</div>
        <div class="metric-sub">Sélectionné parmi 4 modèles</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">F1-Score</div>
        <div class="metric-value">0.686</div>
        <div class="metric-sub">Score principal</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Feature Clé</div>
        <div class="metric-value" style="font-size:1.1rem;margin-top:4px;">Time</div>
        <div class="metric-sub">SHAP = 0.1725</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Dataset</div>
        <div class="metric-value">299</div>
        <div class="metric-sub">Patients · UCI</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Main Columns ──────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="section-title">👤 Profil Patient</div>', unsafe_allow_html=True)
    st.markdown('<div class="patient-card">', unsafe_allow_html=True)

    labels = {
        "age": "Âge", "anaemia": "Anémie", "creatinine_phosphokinase": "CPK",
        "diabetes": "Diabète", "ejection_fraction": "Éjection (%)",
        "high_blood_pressure": "Hypertension", "platelets": "Plaquettes",
        "serum_creatinine": "Créatinine", "serum_sodium": "Sodium",
        "sex": "Sexe", "smoking": "Tabagisme", "time": "Suivi (j)",
        "age_group": "Groupe d'âge", "ef_severity": "Sévérité EF",
        "sodium_risk": "Risque Sodium", "cp_log": "CPK (log)", "platelet_log": "Plaquettes (log)"
    }

    for feat in feature_names:
        val = patient_data[feat].values[0]
        label = labels.get(feat, feat)
        if feat in ["anaemia", "diabetes", "high_blood_pressure", "smoking", "sodium_risk"]:
            display = "⚠️ Oui" if val == 1 else "✅ Non"
        elif feat == "sex":
            display = "👨 Homme" if val == 1 else "👩 Femme"
        elif feat in ["cp_log", "platelet_log"]:
            display = f"{val:.3f}"
        else:
            display = str(val)

        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.04);">
            <span style="font-size:0.8rem;color:#718096;text-transform:uppercase;letter-spacing:0.05em;">{label}</span>
            <span style="font-size:0.9rem;color:#e8e6f0;font-weight:500;">{display}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-title">🔮 Résultat de Prédiction</div>', unsafe_allow_html=True)

    if predict_btn:
        try:
            X_scaled = scaler.transform(patient_data)
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0]

            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-high">
                    <div style="font-family:'DM Serif Display',serif;font-size:1.8rem;color:#fc8181;">⚠️ Risque Élevé</div>
                    <div style="font-size:3.5rem;font-weight:600;color:#dc2626;line-height:1;margin:16px 0 8px 0;">{probability[1]*100:.1f}%</div>
                    <div style="font-size:0.85rem;color:#fc8181;opacity:0.8;">Probabilité de décès par insuffisance cardiaque</div>
                    <div style="margin-top:16px;font-size:0.8rem;color:#9b2c2c;">⚡ Consultation urgente recommandée</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-low">
                    <div style="font-family:'DM Serif Display',serif;font-size:1.8rem;color:#6ee7b7;">✅ Risque Faible</div>
                    <div style="font-size:3.5rem;font-weight:600;color:#10b981;line-height:1;margin:16px 0 8px 0;">{probability[0]*100:.1f}%</div>
                    <div style="font-size:0.85rem;color:#6ee7b7;opacity:0.8;">Probabilité de survie</div>
                    <div style="margin-top:16px;font-size:0.8rem;color:#065f46;">📋 Suivi régulier recommandé</div>
                </div>
                """, unsafe_allow_html=True)

            # SHAP Waterfall
            st.markdown('<div class="graph-card"><div class="graph-title">⚡ Explication SHAP — Ce Patient</div>', unsafe_allow_html=True)
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_scaled)
                fig, ax = plt.subplots(figsize=(7, 4))
                fig.patch.set_facecolor('#13131d')
                ax.set_facecolor('#13131d')
                if isinstance(shap_values, list):
                    sv = shap_values[1][0]
                    ev = explainer.expected_value[1]
                else:
                    sv = shap_values[0]
                    ev = explainer.expected_value
                shap.waterfall_plot(
                    shap.Explanation(values=sv, base_values=ev, data=X_scaled[0], feature_names=feature_names),
                    show=False, max_display=8
                )
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
            except Exception as e:
                st.warning(f"SHAP non disponible : {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Erreur : {e}")

    else:
        st.markdown("""
        <div style="background:#13131d;border:1px dashed rgba(220,38,38,0.2);
                    border-radius:16px;padding:60px 28px;text-align:center;">
            <div style="font-size:3rem;margin-bottom:16px;">🫀</div>
            <div style="font-family:'DM Serif Display',serif;font-size:1.3rem;color:#4a5568;margin-bottom:8px;">Aucune analyse lancée</div>
            <div style="font-size:0.85rem;color:#2d3748;">
                Renseignez les données du patient dans le panneau gauche<br>
                puis cliquez sur <strong style="color:#dc2626;">Analyser le Patient</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── SHAP Global Bars ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Importance des Features — SHAP Global</div>', unsafe_allow_html=True)

shap_features = [
    ("time", 0.1725, "#dc2626"),
    ("serum_creatinine", 0.0974, "#e53e3e"),
    ("ejection_fraction", 0.0612, "#f56565"),
    ("serum_sodium", 0.0412, "#fc8181"),
    ("age", 0.0319, "#feb2b2"),
    ("cp_log", 0.0218, "#4a5568"),
    ("platelet_log", 0.0187, "#4a5568"),
    ("ef_severity", 0.0165, "#4a5568"),
]

bars_html = ""
max_val = shap_features[0][1]
for name, score, color in shap_features:
    pct = (score / max_val) * 100
    bars_html += f"""
    <div class="feature-bar">
        <span class="feature-name">{name}</span>
        <div class="feature-track">
            <div class="feature-fill" style="width:{pct}%;background:linear-gradient(90deg,{color},{color}88);"></div>
        </div>
        <span class="feature-score">{score:.4f}</span>
    </div>
    """
st.markdown(f'<div class="graph-card">{bars_html}</div>', unsafe_allow_html=True)

# ── Tabs Visualisations ───────────────────────────────────────────────────────
st.markdown('<div class="section-title">🖼️ Visualisations</div>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["📈 Évaluation Modèles", "🔬 SHAP Analysis", "📉 Courbes ROC", "🔢 Matrices Confusion"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("notebooks/figures/evaluation_finale.png"):
            st.markdown('<div class="graph-card"><div class="graph-title">Comparaison Finale</div>', unsafe_allow_html=True)
            st.image("notebooks/figures/evaluation_finale.png", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        if os.path.exists("notebooks/figures/metrics_comparison.png"):
            st.markdown('<div class="graph-card"><div class="graph-title">Métriques par Modèle</div>', unsafe_allow_html=True)
            st.image("notebooks/figures/metrics_comparison.png", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("notebooks/figures/shap_importance.png"):
            st.markdown('<div class="graph-card"><div class="graph-title">Importance Globale</div>', unsafe_allow_html=True)
            st.image("notebooks/figures/shap_importance.png", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        if os.path.exists("notebooks/figures/shap_beeswarm.png"):
            st.markdown('<div class="graph-card"><div class="graph-title">SHAP Beeswarm</div>', unsafe_allow_html=True)
            st.image("notebooks/figures/shap_beeswarm.png", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        if os.path.exists("notebooks/figures/shap_summary.png"):
            st.markdown('<div class="graph-card"><div class="graph-title">SHAP Summary</div>', unsafe_allow_html=True)
            st.image("notebooks/figures/shap_summary.png", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        if os.path.exists("notebooks/figures/shap_waterfall.png"):
            st.markdown('<div class="graph-card"><div class="graph-title">SHAP Waterfall</div>', unsafe_allow_html=True)
            st.image("notebooks/figures/shap_waterfall.png", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    if os.path.exists("notebooks/figures/roc_curves.png"):
        st.markdown('<div class="graph-card"><div class="graph-title">Courbes ROC — Tous les Modèles</div>', unsafe_allow_html=True)
        st.image("notebooks/figures/roc_curves.png", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    if os.path.exists("notebooks/figures/confusion_matrices.png"):
        st.markdown('<div class="graph-card"><div class="graph-title">Matrices de Confusion</div>', unsafe_allow_html=True)
        st.image("notebooks/figures/confusion_matrices.png", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:40px 0 20px 0;color:#2d3748;font-size:0.75rem;letter-spacing:0.08em;">
    CARDIOPREDICT AI · CENTRALE CASABLANCA · CODING WEEK 2026 · Random Forest + SHAP
</div>
""", unsafe_allow_html=True)
