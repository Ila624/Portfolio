[README (3).md](https://github.com/user-attachments/files/29143598/README.3.md)
# Predictive Analysis of Diabetes Progression

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ila624/Portfolio/blob/main/Analysis%20of%20Diabetes%20Progression/DiabetesProgression-EN.ipynb)

Predictive study on the progression of diabetes using the Scikit-Learn Diabetes dataset. The project compares three regression models and provides an in-depth interpretation of the results.

---

## Dataset

**Source:** [`sklearn.datasets.load_diabetes`](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)

- 442 patients, 10 clinical variables, all numerical and pre-standardised
- No missing values
- Target: quantitative measure of disease progression one year after baseline (scale: 25–346)

---

## Project Structure

```
├── DiabitesProgression-EN.ipynb   # Main notebook (English)
└── README.md
```

---

## Methodology

1. **Exploratory Data Analysis** — correlation heatmap and scatter plots to identify the most relevant variables (BMI, s5, bp)
2. **Model Comparison** — Ridge, Lasso, and Random Forest evaluated via `Pipeline` + `GridSearchCV` with 5-Fold Cross-Validation
3. **Metric** — R² (Coefficient of Determination), chosen for its interpretability and comparability across models
4. **In-depth Analysis** — coefficient inspection, residual analysis, and Ridge vs Lasso comparison

---

## Results

| Model | Best R² (CV) | Best Parameter |
|---|---|---|
| Ridge | ~0.48 | α ≈ 39 |
| Lasso | ~0.47 | α ≈ 1.93 |
| Random Forest | ~0.40 | max_depth = 5 |

**Winning model: Lasso** — chosen for its automatic feature selection capability despite a marginally lower R² than Ridge.

- **Final R² on test set:** 0.4717
- **RMSE:** ≈ 53 points
- **Variables zeroed out by Lasso:** `age`, `s2` (LDL), `s4` (total cholesterol/HDL)

---

## Key Findings

- **BMI** and **s5** (serum triglycerides) are the strongest positive predictors of disease progression
- Lasso eliminates 3 redundant variables, suggesting a leaner set of clinical indicators is sufficient for monitoring
- The residual error (RMSE ≈ 53) points to the influence of factors not captured in the dataset (e.g. physical activity, diet, genetics)

---

## Notebook

> 📓 [View the full notebook here](DiabitesProgression-EN.ipynb)

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-grey)
![pandas](https://img.shields.io/badge/pandas-grey)
![matplotlib](https://img.shields.io/badge/matplotlib-grey)
![seaborn](https://img.shields.io/badge/seaborn-grey)
