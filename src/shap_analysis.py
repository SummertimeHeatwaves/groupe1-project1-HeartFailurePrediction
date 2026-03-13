import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("models/best_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")
df = pd.read_csv("data/data_processed.csv")

X = df.drop("DEATH_EVENT", axis=1)[feature_names]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, show=False)
plt.savefig("notebooks/figures/shap_summary.png", bbox_inches="tight")
plt.close()

shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.savefig("notebooks/figures/shap_bar.png", bbox_inches="tight")
plt.close()

print("SHAP analysis terminée !")