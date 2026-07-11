# Customer Churn Prediction & Retention Targeting

A tiered project on the [IBM Telco Customer Churn dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
(7,043 customers), built in stages from a basic classifier up to a
business-facing retention-targeting system.

## Why this structure

Most churn portfolio projects stop at "trained a model, got 85% accuracy."
That number is close to meaningless here — the dataset is ~73% "no churn,"
so a model that predicts "no churn" for everyone already scores ~73%
accuracy. This project is built in explicit levels to show the actual
skill progression: from a naive baseline, to correct evaluation, to
explainability, to a genuine business decision layer.

## Notebook

`notebooks/churn_project_walkthrough.ipynb` — a single narrated notebook covering all 6 levels
in one place, already executed (real outputs, real SHAP chart, real numbers baked in).

## Levels

| Level | File | What it adds |
|---|---|---|
| 1 | `src/level1_basic_model.py` | Baseline LogReg / Random Forest, accuracy metric only — and the naive-baseline comparison that shows why accuracy is misleading here |
| 2 | `src/level2_improved_eval.py` | Precision/recall/F1/ROC-AUC/PR-AUC, 5-fold CV, engineered features (tenure buckets, service adoption count, contract commitment, high-risk interaction flag), XGBoost benchmark |
| 3 | `src/level3_explainability.py` | SHAP values for true churn drivers (direction + magnitude, not just feature_importances_) |
| 4 | `src/level4_cost_sensitive.py` | Cost-sensitive threshold optimization — turns "84% AUC" into a dollar figure: optimized threshold captures ~40% more expected value than the default 0.5 cutoff |
| 5 | `src/level5_uplift_modeling.py` | T-learner uplift model on a **documented synthetic retention-offer experiment** (see note below) — separates "persuadable" customers from "lost causes" and "sure things"; uplift-based targeting captures substantially more value than risk-based targeting at the same budget |
| 6 | `app/streamlit_app.py` | Interactive app: portfolio overview, per-customer SHAP explanation, business-impact threshold curve |

## Important honesty note on Level 5 (uplift modeling)

No public churn dataset includes real "did we offer this customer a
retention deal" data, because that requires a company's own randomized
experiment. To demonstrate uplift methodology, Level 5 **simulates** a
retention-offer experiment on top of the real customer features: treatment
is randomly assigned (like an RCT), and the treatment effect is a known
function I planted (varies by contract type and internet service, plus
noise). This lets the project both fit an uplift model *and* verify it
recovers the true planted effect — a validation step real-world uplift
projects usually can't do.

## Key results

- **Level 1 → 2**: Logistic Regression with engineered features hit
  ROC-AUC 0.847 / PR-AUC 0.662 — and notably, plain LogReg edged out
  XGBoost once feature engineering was in place, worth mentioning as a
  "let the data decide, don't default to the fanciest model" story.
- **Level 3**: Contract type (month-to-month), tenure, and lack of online
  security were the top churn drivers by SHAP value.
- **Level 4**: Optimizing the decision threshold (0.15 vs. default 0.5)
  captured ~40% more expected retention value on the same model.
- **Level 5**: Uplift-based targeting captured ~$28K more expected value
  than risk-based targeting at the same outreach budget, by correctly
  skipping high-risk "lost cause" customers a discount wouldn't save.

## Running it

```bash
pip install -r requirements.txt

python src/level1_basic_model.py
python src/level2_improved_eval.py
python src/level3_explainability.py
python src/level4_cost_sensitive.py
python src/level5_uplift_modeling.py

streamlit run app/streamlit_app.py
```

## Stack

Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit.
