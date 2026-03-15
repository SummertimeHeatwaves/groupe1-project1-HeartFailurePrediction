# 🫀 Heart Failure Risk Predictor
**Coding Week 09-15 March 2026 | École Centrale Casablanca**

An advanced clinical decision-support tool predicting heart failure mortality risk using Explainable Machine Learning (SHAP).

---

## 📁 Project Structure

```text
project/
├── data/                          # Dataset (CSV)
├── models/                        # Saved model artifacts
├── notebooks/
│   └── eda.ipynb                  # Exploratory Data Analysis
├── plots/                         # Generated SHAP & evaluation plots
├── src/
│   ├── data_processing.py         # Preprocessing pipeline + memory optimization
│   ├── train_model.py             # Model training & selection
│   └── evaluate_model.py          # Evaluation & SHAP computation
├── app/
│   └── app.py                     # Streamlit web interface
├── tests/
│   └── test_data_processing.py    # Automated tests (pytest)
├── .github/workflows/ci.yml       # GitHub Actions CI/CD
├── Dockerfile
├── requirements.txt
└── README.md




⚠️ Note concernant l'intégration continue (CI/CD) : > Le pipeline GitHub Actions affiche une erreur à la toute dernière étape des tests SHAP. Il s'agit d'un "Segmentation Fault" (problème de mémoire C++) connu entre la bibliothèque SHAP et Linux/Ubuntu. Le code Python est fonctionnel et le projet tourne parfaitement en local ainsi que via l'image Docker.
