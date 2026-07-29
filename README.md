# Machine Learning-Based Incident Prediction in Distributed Systems Using Observability Data

> A Machine Learning framework for proactive incident prediction in distributed systems using synthetic observability metrics.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b)

---

## Overview

Modern cloud-native applications generate enormous volumes of observability data such as CPU utilization, memory usage, latency, throughput and error rates. Traditional monitoring systems rely on threshold-based alerting, detecting problems only after service degradation has already occurred.

This project proposes a Machine Learning-based framework that predicts incidents proactively using observability metrics collected from distributed microservices.

The project was developed as part of the **M.Tech (Artificial Intelligence & Machine Learning)** dissertation at **BITS Pilani**.

---

## Objectives

- Generate a synthetic observability dataset
- Engineer temporal features for incident prediction
- Compare multiple Machine Learning models
- Predict incidents before failures occur
- Visualize predictions using an interactive dashboard

---

## Features

- Synthetic observability dataset generation
- Feature engineering
- Incident labeling
- Logistic Regression
- Random Forest
- XGBoost
- LSTM
- Model comparison
- ROC Curve generation
- Confusion Matrix visualization
- Interactive Streamlit dashboard

---

## Project Architecture

```text
Synthetic Dataset Generation
            │
            ▼
Observability Metrics Collection
            │
            ▼
Data Preprocessing
            │
            ▼
Feature Engineering
            │
            ▼
Incident Label Generation
            │
            ▼
Machine Learning Models
            │
            ▼
Performance Evaluation
            │
            ▼
Interactive Streamlit Dashboard
```

---

## Dataset

The synthetic dataset simulates observability metrics collected from five distributed microservices.

### Dataset Summary

| Property | Value |
|----------|-------|
| Services | 5 |
| Monitoring Duration | 30 Days |
| Sampling Interval | 1 Minute |
| Total Records | 216,000 |
| Simulated Incidents | 80 |

### Metrics

- CPU Utilization
- Memory Utilization
- Response Latency
- Throughput
- Error Rate

---

## Feature Engineering

The following temporal features are generated for each observability metric:

- Rolling Mean
- Rolling Standard Deviation
- Lag Features
- Delta Features

Prediction targets are created for:

- Target_5min
- Target_10min
- Target_15min

---

## Machine Learning Models

| Model | Purpose |
|--------|----------|
| Logistic Regression | Baseline classification model |
| Random Forest | Ensemble learning model |
| XGBoost | Gradient boosting model |
| LSTM | Deep learning model for sequential prediction |

---

## Evaluation Metrics

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

## Dashboard Features

The Streamlit dashboard provides:

- Interactive visualization
- Incident timeline
- Model comparison
- Operational risk monitoring
- Prediction analysis

---

## Project Structure

```text
AIOps-IncidentPrediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── dataset.py
│   ├── feature_engineering.py
│   ├── labeling.py
│   ├── model_comparison.py
│   ├── dashboard/
│   └── models/
│
├── results/
│   ├── metrics/
│   ├── comparison/
│   ├── confusion_matrices/
│   └── roc_curves/
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/koel2309-wq/AIOps-IncidentPrediction.git

cd AIOps-IncidentPrediction
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Generate Dataset

```bash
python src/dataset.py
```

Feature Engineering

```bash
python src/feature_engineering.py
```

Generate Labels

```bash
python src/labeling.py
```

Train Models

```bash
python src/models/logistic_regression.py

python src/models/random_forest.py

python src/models/xgboost.py

python src/models/lstm.py
```

Compare Models

```bash
python src/model_comparison.py
```

Launch Dashboard

```bash
streamlit run app.py
```

---

## Results

- Synthetic observability dataset successfully generated.
- Four Machine Learning models implemented and evaluated.
- LSTM achieved the best overall predictive performance.
- Interactive Streamlit dashboard demonstrated practical applicability.

---

## Future Work

- Real-time streaming prediction
- Explainable AI (SHAP/LIME)
- Automated incident remediation
- Integration with Prometheus and Grafana
- Validation using production observability datasets

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- PyTorch
- Matplotlib
- Streamlit

---

## Author

**Koel Banerjee**

M.Tech (Artificial Intelligence & Machine Learning)

BITS Pilani

---

## License

This project was developed for academic and research purposes as part of the M.Tech dissertation at BITS Pilani.