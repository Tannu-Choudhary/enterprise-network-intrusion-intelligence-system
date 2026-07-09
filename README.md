# Enterprise Network Intrusion Intelligence System

An end-to-end machine learning system for detecting and classifying network intrusions in enterprise traffic, built on the CICIDS2017 dataset. The system covers the full pipeline — data preprocessing, feature engineering, model training and evaluation, and an interactive analyst dashboard — for identifying benign vs. malicious network flows and classifying attack types in near real time.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Team](#team)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

Enterprise networks generate massive volumes of traffic, within which malicious activity — port scans, DoS/DDoS attacks, brute-force attempts, botnets, and infiltration — must be identified quickly and accurately. This project builds a supervised machine learning system that:

- Cleans and preprocesses raw network flow data (CICIDS2017)
- Engineers and selects the most informative traffic features
- Trains and evaluates multiple classification models to distinguish benign traffic from various attack categories
- Serves predictions and analytics through an interactive Streamlit dashboard for security analysts

This is an academic project developed by a team of three, with responsibilities divided across data/feature engineering, modeling/inference, and dashboard development.

## Dataset

**CICIDS2017 (Cleaned & Preprocessed)**
Source: [Kaggle — ericanacletoribeiro/cicids2017-cleaned-and-preprocessed](https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/data)

The dataset contains labeled network flow records capturing both benign traffic and a range of attack types (DoS, DDoS, Port Scan, Brute Force, Web Attacks, Infiltration, Botnet, and others), with over 80 flow-based features derived using CICFlowMeter.

> Place raw dataset files in `data/raw/` before running the preprocessing pipeline. Processed outputs are written to `data/processed/`.

## Project Structure

```
enterprise-network-intrusion-intelligence-system/
├── data/
│   ├── raw/                     # Original, unmodified dataset files
│   └── processed/                # Cleaned, encoded, and engineered datasets
│
├── models/
│   ├── best_model.pkl             # Serialized best-performing trained model
│   ├── scaler.pkl                 # Fitted feature scaler
│   ├── label_encoder.pkl          # Fitted target label encoder
│   └── selected_features.csv      # Final selected feature set
│
├── reports/
│   ├── figures/                   # EDA and evaluation plots
│   ├── metrics/                   # Saved evaluation metrics
│   └── screenshots/               # Dashboard/application screenshots
│
├── docs/
│   ├── diagrams/                  # Architecture and workflow diagrams
│   └── presentation/              # Project presentation materials
│
├── src/
│   ├── data/
│   │   └── data_preprocessing.py       # Data cleaning, encoding, splitting, scaling
│   ├── features/
│   │   └── feature_engineering.py      # Feature selection & dimensionality reduction
│   ├── visualization/
│   │   ├── eda.py                      # Exploratory data analysis
│   │   └── plots.py                    # Reusable evaluation/plotting library
│   │
│   ├── models/
│   │   └── model_trainer.py            # Model training & hyperparameter tuning
│   ├── inference/
│   │   └── predictor.py                # Inference/prediction pipeline
│   ├── train_model.py                  # Training entry point
│   ├── evaluate_model.py               # Evaluation entry point
│   │
│   ├── dashboard/
│   │   ├── app.py                      # Streamlit dashboard entry point
│   │   ├── home.py                     # Dashboard home page
│   │   ├── prediction.py               # Live prediction interface
│   │   ├── analytics.py                # Analytics & visualization page
│   │   └── about.py                    # Project/about page
│   │
│   └── utils/
│       ├── config.py                   # Centralized project configuration
│       ├── logger.py                   # Shared logging utility
│       └── helper.py                   # General-purpose shared helpers
│
├── README.md
├── requirements.txt
├── main.py
└── CLAUDE.md
```

## Installation

**Prerequisites:** Python 3.11+

1. Clone the repository and navigate into the project directory:
   ```bash
   git clone  https://github.com/Tannu-Choudhary/enterprise-network-intrusion-intelligence-system.git
   cd enterprise-network-intrusion-intelligence-system
   ```

2. Create and activate a virtual environment:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate       # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the [CICIDS2017 (Cleaned & Preprocessed)](https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/data) dataset from Kaggle and place the CSV file(s) in `data/raw/`.

## Usage

Run the full pipeline via the main entry point:

```bash
python main.py --stage all
```

Or run individual stages:

```bash
# Preprocess raw data
python -m src.data.data_preprocessing --raw-data data/raw/cicids2017.csv

# Engineer and select features
python -m src.features.feature_engineering

# Generate EDA report
python -m src.visualization.eda --input data/raw/cicids2017.csv

# Train models
python src/train_model.py

# Evaluate the trained model
python src/evaluate_model.py

# Launch the analyst dashboard
streamlit run src/dashboard/app.py
```

## Pipeline Stages

| Stage | Module | Description |
|---|---|---|
| 1. Preprocessing | `src/data/data_preprocessing.py` | Cleans raw traffic data, handles missing/infinite values, encodes labels, splits and scales data |
| 2. Feature Engineering | `src/features/feature_engineering.py` | Removes low-variance/redundant features, ranks importance, selects final feature set |
| 3. EDA | `src/visualization/eda.py` | Generates dataset overview, class distribution, correlation, and feature distribution reports |
| 4. Training | `src/train_model.py`, `src/models/model_trainer.py` | Trains and tunes candidate classification models |
| 5. Evaluation | `src/evaluate_model.py` | Evaluates trained model(s) using confusion matrices, ROC/PR curves, and metrics |
| 6. Inference | `src/inference/predictor.py` | Loads the trained model to classify new network flow samples |
| 7. Dashboard | `src/dashboard/` | Interactive Streamlit app for predictions and analytics |

## Team

| Member | Responsibility |
|---|---|
| Member A | Data preprocessing, feature engineering, and visualization/EDA |
| Member B | Model training, evaluation, and inference |
| Member C | Interactive analyst dashboard |

## Tech Stack

- **Language:** Python 3.11
- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn, joblib
- **Visualization:** matplotlib, seaborn, plotly
- **Dashboard:** streamlit

## License

This project is intended for academic and educational purposes.