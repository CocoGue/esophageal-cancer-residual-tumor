# Residual Disease Prediction in Esophageal Cancer
Official repo of the "Machine learning based detection of residual esophageal tumor after neoadjuvant chemoradiotherapy using post-treatment CT and clinical parameters" *Under review*

This repository provides a clean, reproducible pipeline to train and apply a logistic regression model for predicting residual disease in esophageal cancer patients based on clinical and imaging features.

A **command-line interface (CLI)** is provided for both training and inference.

---

### How to use it?

1. Clone the repository

```bash
git clone https://github.com/CocoGue/esophageal-cancer-residual-tumor.git
cd esophageal-cancer-residual-tumor
```

2. Create and activate a virtual environment. Install the dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

3. Command-Line interface

You can train the a logistic regression with:
```bash
python -m ct_residual_disease.main train data/toy_dataset_training.csv \
  --feature-mode combined \
  --scale-numerical \
  --drop-first \
  --add-intercept \
  --elasticnet
```

You can use trained logistic regression on new data with:
```bash
python -m ct_residual_disease.main infer data/toy_dataset_test.csv \
  --feature-mode combined \
  --drop-first
```

---
Input data format

The input CSV must contain the following columns:

| Column              | Description                                                |
|---------------------|------------------------------------------------------------|
| ID                  | Patient identifier (e.g. ID_01)                            |
| Study_ID            | Numeric ID extracted from ID                               |
| Age                 | Age (years)                                                |
| Gender              | Male / Female                                              |
| Histology           | Adenocarcinoma / Squamous cell carcinoma                   |
| cT                  | T2 / T3 / T4                                               |
| cN                  | N0 / N1 / N2 / N3                                          |
| Volume_Component    | Tumor volume                                               |
| Residual_Disease    | Target (0 = no, 1 = yes) 
