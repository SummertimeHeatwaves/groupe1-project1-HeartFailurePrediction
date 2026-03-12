# Heart Failure Prediction Project ❤️📊

## Overview
This project aims to predict patient survival (DEATH_EVENT) using clinical records. By applying machine learning models (Random Forest, XGBoost) and handling class imbalances, we are building a predictive tool to assist in medical data analysis.

## Project Phases
- [x] **Phase 1: Exploratory Data Analysis (EDA)** - Verified 0 missing values in the dataset.
  - Analyzed outliers using boxplots and the IQR method.
- [x] **Phase 2: Data Cleansing**
  - Applied capping to extreme outliers (e.g., creatinine_phosphokinase) to preserve medical significance while improving model stability.
  - Created `cleaned_dataset.csv`.
- [ ] **Phase 3: Model Training** (In Progress)
- [ ] **Phase 4: Web Application**

## How to Run This Project
1. Clone the repository: `git clone https://github.com/SummertimeHeatwaves/groupe1-project1-HeartFailurePrediction.git`
2. Navigate to the folder: `cd groupe1-project1-HeartFailurePrediction`
3. Ensure you have the dataset in the `data/` folder.
4. Run the notebooks in the `notebooks/` directory to see the EDA and modeling steps.
