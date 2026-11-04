[README.md](https://github.com/user-attachments/files/28757524/README.md)
# Titanic Passenger Analysis and Predictions

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ila624/Portfolio/blob/main/Titanic%20Analysis/TitanicAnalysis_(EN).ipynb)

A machine learning project to predict the survival of Titanic passengers, built with a rigorous methodology focused on result generalization and model interpretability.

## Overview

This project analyzes the Titanic passenger dataset and builds a **Decision Tree** classifier to predict whether a passenger survived. The choice of model is motivated by the nature of the problem: rescue decisions on the Titanic followed clear hierarchical rules ("women and children first") that a Decision Tree naturally reflects.

The notebook covers the full ML workflow: exploratory data analysis, preprocessing pipeline, depth optimization, model evaluation, and comparison against baseline models.

## Results

| Model | Accuracy |
|---|---|
| Decision Tree | **81.17%** |
| KNN | 80.27% |
| Logistic Regression | 79.37% |

Beyond accuracy, the Decision Tree stands out for its **Precision on the Survived class (86%)**, meaning very few false positives — passengers incorrectly predicted as survivors.

## Project Structure

```
├── TitanicAnalysis_EN.ipynb   # Main notebook
├── titanic_sub_2.csv          # Dataset
└── README.md
```

## Dataset

The dataset includes the following features:

| Field | Description |
|---|---|
| `Survived` | Target variable (0 = No, 1 = Yes) |
| `Pclass` | Passenger class (1st, 2nd, 3rd) |
| `Sex` | Gender |
| `Age` | Age |
| `Embarked` | Port of embarkation (S, C, Q) |

Fields excluded from the analysis: `Fare` (multicollinearity with `Pclass`), `SibSp` and `Parch` (skewed distribution), `Name` and `Ticket` (high cardinality).

## Methodology

- **Train/Validation/Test split** (75:25 ratio, applied twice) to prevent data leakage
- **Scikit-learn Pipeline** for automated preprocessing: median imputation for numerical features, mode imputation + One-Hot Encoding for categorical ones
- **Depth optimization** via validation set to find the best `max_depth` and avoid overfitting
- **Baseline comparison** against Logistic Regression and KNN, both trained with the same pipeline and scaling

## Requirements

```
numpy
pandas
scipy
matplotlib
scikit-learn
```

Install with:
```bash
pip install numpy pandas scipy matplotlib scikit-learn
```
