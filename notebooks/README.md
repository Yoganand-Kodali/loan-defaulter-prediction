## Notebooks

Run the notebooks in the following order for end-to-end reproducibility.

---

### 1. `CAPSTONE_EDA_FINAL.ipynb` — EDA on 75-Column Dataset
- Loads `application_data.csv` (raw Kaggle data)
- Handles missing values for the subset of rows where `HOUSETYPE_MODE` is null (75-column path)
- Explores each feature's relationship with the TARGET variable
- Feature engineering: grouping categories, transformations (log, cube root), encoding
- Outputs cleaned dataset for modelling

**Key sections:** Column dropping → Feature-by-feature EDA → Statistical tests → KNN imputation

---

### 2. `CAPSTONE_EDA_FOR_122_COL.ipynb` — EDA on Full 122-Column Dataset
- Handles the full dataset including 50 apartment/building columns
- Covers building-related features (APARTMENTS_MEDI, BASEMENTAREA_MEDI, etc.)
- Same EDA pattern as above but for the 122-column path
- Outputs `df122_iterative.csv` after preprocessing

**Key sections:** Building feature analysis → OHE → KNN imputation in batches

---

### 3. `Modelling.ipynb` — Baseline Experiments
- Loads cleaned data (`after_knn.csv`)
- Establishes baseline with Logistic Regression and Decision Tree (no SMOTE)
- Shows the imbalance problem clearly (F1 ≈ 0 for minority class)
- GridSearchCV for Decision Tree depth tuning
- Initial Random Forest experiments

**Purpose:** Demonstrates *why* SMOTE is necessary before full modelling.

---

### 4. `capstone_75_model.ipynb` — Full Modelling on 75-Column Dataset
- Loads `df_75.csv` (preprocessed 75-column data)
- SMOTE with `sampling_strategy=0.25` (4:1 ratio)
- RFE for feature selection (top 30 features)
- Trains all 7 models: LR, DT, RF, GradientBoosting, XGBoost, AdaBoost, Stacking
- Reports F1 scores with and without RFE

---

### 5. `122.ipynb` — Full Modelling on 122-Column Dataset
- Loads `data122_unscaled_knn_5.csv` (preprocessed 122-column data)
- SMOTE with `sampling_strategy=1` (1:1 ratio) and `k_neighbors=5`
- RFE with 30 selected features
- Same 7 models as above
- **Best result:** Gradient Boosting → Train F1: 0.933, Test F1: 0.932

---

### Dependencies
See `../requirements.txt` for all Python package requirements.
