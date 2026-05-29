## Data Directory

This folder holds the processed dataset used for modelling.

### Files

| File | Description | Size |
|------|-------------|------|
| `df122_iterative.csv` | Preprocessed 122-column dataset after KNN imputation, encoding, and feature engineering | ~161 MB |

### How to get the raw data

1. Download `application_data.csv` from [Kaggle — Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)
2. Place it in this `data/` folder
3. Run the EDA notebooks to reproduce the preprocessing steps

> **Note:** Large CSV files are excluded from git tracking via `.gitignore`. Place them here locally after cloning.
