# Medical Insurance Cost Prediction (No Notebooks)

This project predicts **medical insurance charges** using demographic + lifestyle features (age, BMI, smoking, etc.).
It is written as a **regular Python project** (no Jupyter / Colab) so it runs nicely on GitHub.

## Project flow

1. Data cleaning + preprocessing  
2. EDA (plots saved to `reports/figures/`)  
3. Feature engineering (BMI category, interactions)  
4. Train and compare models (Linear, Ridge, Lasso, Random Forest, Gradient Boosting)  
5. Evaluate with MAE / MSE / RMSE / R²  
6. Interpret results (feature importance / coefficients)

## Folder structure

```
medical_insurance_cost_prediction/
├── data/
│   ├── raw/insurance.csv
│   └── processed/
├── models/                       # saved trained models
├── reports/
│   ├── figures/                  # plots from EDA + evaluation
│   └── metrics/                  # metrics CSV files
├── src/                          # main project code
├── tests/                        # small unit tests
├── requirements.txt
└── run.py                        # easy entry script
```

## Setup (local)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run EDA (creates plots)

```bash
python run.py eda
```

## Train models (saves best model to `models/`)

```bash
python run.py train
```

## Evaluate best saved model (creates metrics + plots)

```bash
python run.py eval
```

## Notes for GitHub

- Your professor said “it must run on GitHub”:  
  That usually means the repository should include a normal Python structure + clear commands.
- This repo includes a simple GitHub Actions workflow in `.github/workflows/ci.yml`.
  It **runs tests** and a quick “import smoke check”.
- The dataset is stored under `data/raw/insurance.csv`.  
  If your professor prefers not to commit data, you can remove it and add instructions in README.

## Example outputs

After running:
- `reports/figures/charges_distribution.png`
- `reports/metrics/model_comparison.csv`
- `models/best_model.joblib`

