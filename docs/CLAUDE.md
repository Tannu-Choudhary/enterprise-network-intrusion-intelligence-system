# CLAUDE.md

## Project Name
Enterprise Network Intrusion Intelligence System

---

## Project Overview

This project is an AI-powered Enterprise Network Intrusion Intelligence System that detects and classifies malicious network traffic using Machine Learning.

The project uses the CIC-IDS2017 dataset and aims to build a modular, production-quality intrusion detection pipeline.

---

## Dataset

Dataset Name:
CIC-IDS2017

Location:
data/raw/

Processed dataset:
data/processed/

---

## Technology Stack

Programming Language:
- Python 3.11+

Libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly
- streamlit
- joblib

Development Tools:
- VS Code
- Git
- GitHub
- Claude AI
- ChatGPT

---

## Project Structure

src/
    data/
        data_preprocessing.py

    features/
        feature_engineering.py

    utils/

    visualization/

    train_model.py

    evaluate_model.py

data/
    raw/

    processed/

docs/

reports/

tests/

---

## Coding Standards

- Follow PEP 8.
- Write clean and readable code.
- Use descriptive variable names.
- Use functions instead of long scripts.
- Every function must include a docstring.
- Add comments only where necessary.
- Avoid duplicate code.

---

## Error Handling

Always use try-except blocks when reading files or saving data.

Display meaningful error messages.

Never allow the program to crash because of simple file errors.

---

## Logging

Use Python logging instead of print() for important operations.

Log:
- Dataset loaded
- Dataset shape
- Missing values handled
- Duplicate rows removed
- Model trained
- Model saved

---

## Data Processing Rules

- Handle missing values.
- Remove duplicates.
- Remove unnecessary columns.
- Encode categorical features.
- Scale numerical features when required.
- Save processed dataset.

---

## Machine Learning Rules

Train multiple models.

Possible models:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost (optional)

Compare all models before selecting the best one.

Save only the best-performing model.

---

## Code Quality

Code should be:

- Modular
- Reusable
- Easy to understand
- Well documented

Avoid writing everything inside one file.

---

## Git Rules

Write meaningful commit messages.

Examples:

git commit -m "Implement data preprocessing pipeline"

git commit -m "Add feature engineering module"

git commit -m "Train Random Forest model"

---

## Documentation

Every Python file must begin with a module docstring explaining its purpose.

Every function must include:

- Description
- Parameters
- Returns

---

## Goal

Build an industry-style AI-based Network Intrusion Detection System that is modular, maintainable, and suitable for academic projects and professional portfolios.