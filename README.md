---
title: Credit Risk Defaulter Prediction App
emoji: 📊
colorFrom: blue
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
---

# Credit Risk Defaulter Prediction App

Deploys a LightGBM credit risk model behind a Gradio web interface. Users upload CSV files and receive default predictions with probabilities for each row.

**Live App:** [Hugging Face Space](https://huggingface.co/spaces/MIKE20Z/credit-risk-prediction)

## Project Structure
```
├── app.py                          # Gradio app — model loading, validation, inference
├── lgbm_credit_risk_model.joblib   # Trained LightGBM classifier (Module 8)
├── lgbm_credit_risk_features.joblib# 23 expected feature names
├── requirements.txt                # Python dependencies
├── files/
│   ├── sample_input.csv            # 5-row example (mixed defaults/non-defaults)
│   └── sample_small.csv            # 2-row example
├── tests/
│   └── test_app.py                 # Model loading and prediction smoke tests
└── .github/workflows/
    ├── ci.yml                      # Lint + test on push/PR to main
    └── cd.yml                      # Deploy to Hugging Face Space on push to main
``` 

## Run Locally
```bash
pip install -r requirements.txt
python app.py
```

## Run Tests

Run from project root:
```bash
pytest tests/test_app.py --maxfail=1
```

## CI/CD

**CI** — triggers on push/PR to main. Checks out repo with LFS, installs dependencies, runs flake8 linting, runs pytest.

**CD** — triggers on push to main. Uploads repo to Hugging Face Space using `hf upload` via `huggingface_hub` CLI. Requires `HF_USERNAME` and `HF_TOKEN` as repository secrets.

## Dataset

UCI Default of Credit Card Clients (30,000 records, 23 features). Model trained with balanced class weights in Module 8 of AAI6610.

## Sample Test sets
Small sample tests sets are include so the user can see how the applications functions and executes


