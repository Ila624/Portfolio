# Supermarket Sales & Apple Quality: ML Case Studies

End-to-end machine learning analysis covering four core problem types — regression, time series forecasting, classification, and clustering — applied to two independent retail-related datasets.

## Overview

The project is split into two self-contained parts:

**Part 1 — Supermarket Sales.** Using transaction-level data from three supermarket branches, the analysis builds regression models (Linear, Polynomial, Random Forest) to predict a transaction's Rating, then shifts to time series forecasting (Linear Regression, XGBoost) to test whether gross income can be predicted over time. The investigation also includes a category-level breakdown of profit.

**Part 2 — Apple Quality.** Using physical and organoleptic measurements of 4,000 apples, the analysis applies supervised classification (Logistic Regression, Decision Tree) to predict quality labels, followed by unsupervised K-Means clustering to test whether quality groupings emerge naturally from the data without labels.

Each part includes data exploration, preprocessing, model training, evaluation, and a discussion of results grounded in the data rather than just metric scores — for example, identifying that gross income is an arithmetic byproduct of other columns rather than an independently observable quantity, or that bad-apple recall improves with a Decision Tree at the cost of more good apples being discarded.

## Repository Contents

| File | Description |
|---|---|
| [VenditeSupermercato_IT](./VenditeSupermercato_IT) | Italian version of the notebook |
| [SupermarketSales_EN](./SupermarketSales_EN) | English version of the notebook |
| [apple_quality](./apple_quality.csv) | Dataset analyzed in part 2 |
| [supermarket_sales](./supermarket_sales.csv) | Dataset analyzed in part 1 |

## Datasets

- **supermarket_sales** — 1,000 retail transactions across three store branches, including product line, payment method, unit price, quantity, and gross income.
- **apple_quality** — 4,000 labeled samples with physical attributes (size, weight, sweetness, crunchiness, juiciness, ripeness, acidity).

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, SciPy, Matplotlib, Seaborn

## Key Takeaways

- Linear Regression slightly outperformed Polynomial Regression and Random Forest in predicting transaction Rating, suggesting the underlying relationship is closer to linear given the mostly categorical features.
- Gross income showed no meaningful temporal pattern — both Linear Regression and XGBoost produced near-zero or negative R² scores, pointing to external factors (promotions, weather, competition) not captured in the dataset.
- A Decision Tree improved recall on detecting bad apples compared to Logistic Regression (0.82 vs. lower), reducing false positives at the cost of more good apples being misclassified — a trade-off relevant to real-world quality control.
- K-Means clustering on apple features did not cleanly separate into two quality-based groups; a third cluster improved separation, hinting at an intermediate quality category.
