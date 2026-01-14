# ***🧠 DEA–Ensemble Learning Framework for Efficiency Prediction***
This repository provides the full Python implementation accompanying the paper:
_"Kehinde, T. O., Oyedele, A. A., Kareem, M. K., Akpan, J., & Olanrewaju, O. A. (2026). Explainable DEA–ensemble approach with golden jackal optimization: efficiency evaluation and prediction for United States information technology firms. Machine Learning with Applications, 23, 100798. https://doi.org/https://doi.org/10.1016/j.mlwa.2025.100798"_


![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Ensemble-green)
![Optimization](https://img.shields.io/badge/Optimization-GJO-orange)
![Status](https://img.shields.io/badge/Status-Published-success)


## ***The code implements an end-to-end (DEA–Machine Learning) framework that integrates:***

* ✅Data Envelopment Analysis (DEA)

* ✅Ensemble learning models

* ✅Golden Jackal Optimization (GJO) for hyperparameter tuning

* ✅Explainable AI (XAI) using SHAP

## ***1. Overview***

Traditional DEA provides static efficiency benchmarking but lacks predictive capability and scalability.
This framework extends DEA by training ensemble machine learning models to predict DEA efficiency scores, enabling scalable, forward-looking efficiency analysis.

The pipeline consists of:

    1. DEA efficiency estimation (CCR / BCC models)

    2. Robustness testing under controlled noise

    3. Predictive modeling using six ensemble regressors

    4. Metaheuristic optimization with Golden Jackal Optimization (GJO)

    5. Explainability analysis via feature importance and SHAP

The framework is designed for large-scale financial datasets and demonstrated on 3,940 DMUs (2013–2023).

## ***2. Implemented Models***

The following ensemble regressors are implemented and benchmarked:

- 🌲 Random Forest Regressor (RF)
- 🌳 Extra Trees Regressor (ET)
- 🚀 XGBoost Regressor (XGB)
- 🌟 LightGBM Regressor (LGBM)
- 🔁 Gradient Boosting Regressor (GBR)
- ➕ AdaBoost Regressor (ADA)


Each model:

* ✅Is trained on DEA-generated efficiency scores

* ✅Uses GJO for hyperparameter optimization

* ✅Produces interpretable outputs (feature importance + SHAP)

## ***3. Project Structure***

```
dea_ensemble_gjo/
├── 📁 data/
│   ├── 📁 raw/                    # Put your dataset.xlsx here
│   └── 📁 processed/              # DEA results will be saved here (auto-created)
├── 📁 src/
│   ├── __init__.py
│   ├── dea_model.py           # DEA analysis
│   ├── xgb_model.py           # XGBoost
│   ├── gbr_model.py           # Gradient Boosting
│   ├── lgb_model.py           # LightGBM
│   ├── rf_model.py            # Random Forest
│   ├── et_model.py            # Extra Trees
│   ├── ada_model.py           # AdaBoost
│   ├── run_pipeline.py        # Main pipeline 
│   └── compare_models.py      # Model comparison script
├── 📁 outputs/                   # All model results (auto-created)
├── 📁 comparison_results/        # Comparison reports (auto-created)
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```



## ***4. Data Description***

* Domain: United States Information Technology firms

* Source: WRDS

* Period: 2013–2023

* Sample size: 3,940 firm-year observations

DEA Inputs

    1. Total Assets

    2. Total Equity

    3. Operating Expenses

DEA Outputs

    1. Revenue

    2. Net Income

    3. Operating Income

DEA efficiency scores (CCR and BCC) are computed first and used as target labels for all ML models.

## ***5. Experimental Design***

    * ✅Train/Test split: 85% / 15%

    * ✅Cross-validation: 5-fold CV (training set only)

    * ✅Optimization objective: Minimize RMSE

    * ✅Optimization method: Golden Jackal Optimization (GJO)

    * ✅Explainability:

    * ✅Embedded feature importance (tree-based)

    * ✅SHAP (TreeSHAP / KernelSHAP)

## ***6. Running the Code***
    Step 1: Install dependencies
    -   pip install -r requirements.txt

    Step 2: Run the full pipeline
    -   python src/run_pipeline.py

This will:

    * ✅Load DEA results

    * ✅Optimize and train all ensemble models

    * ✅Generate predictions, metrics, and explainability outputs

    * ✅Save all results to the outputs/ directory

    * ✅Optional: Model comparison: python src/compare_models.py

## ***7. Outputs***

Each model run produces:

    1. Optimized hyperparameters

    2. Train/Test performance metrics (RMSE, MAE, MAPE, sMAPE, R², R)

    3. Prediction files (CSV)

    4. Feature importance plots

    5. SHAP summary and dependence plots

    6. Serialized trained models (.pkl)

## ***8. Citation***

If you use, modify, adopt any part of this code or adopt its framework in anyway, please cite:

    Kehinde, T. O., Oyedele, A. A., Kareem, M. K., Akpan, J., & Olanrewaju, O. A. (2026). Explainable DEA–ensemble approach with golden jackal optimization: efficiency evaluation and prediction for United States information technology firms. Machine Learning with Applications, 23, 100798. https://doi.org/https://doi.org/10.1016/j.mlwa.2025.100798


## ***9. License***

This code is released for academic and research purposes. 
Refer to the journal article for licensing and reuse conditions.