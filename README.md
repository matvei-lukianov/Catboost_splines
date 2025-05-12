# Predicting Bitcoin Price Movements Using CatBoost and Logistic Regression with Splines

This repository contains the full implementation and analysis from the AMS 515 project focused on predicting short-term Bitcoin price movements using order book data. The project compares two machine learning approaches: **CatBoost**, a gradient boosting model, and **Logistic Regression with Cubic Splines**, a statistically interpretable model.

## Repository Contents

- `Case_report.pdf`: Final project report documenting methodology, indicators, results, and conclusions.
- `README.md`: This file.
- `orderbook.py`: Script for feature extraction and indicator computation from raw limit order book data.
- `project.ipynb`: Jupyter notebook with full modeling pipeline, including EDA, model training, evaluation, and visualization.
- `run.py`: Script to execute the main training and evaluation workflow.

## Project Overview

### Objective
To build and compare models that classify whether the Bitcoin mid-price will move up within the next second, using engineered order book features.

### Models Compared
- **Logistic Regression with Cubic Splines**: For interpretability and smooth feature effects.
- **CatBoost**: For flexible, high-performance modeling of nonlinear interactions.

### Key Features
- Volume imbalances
- Log price statistics
- Quantile-based order book stripe indicators
- Spline transformations

## Results Summary

| Model                  | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|------------------------|----------|-----------|--------|----------|---------|
| Logistic + Splines     | 0.65     | 0.59      | 0.39   | 0.47     | 0.67    |
| CatBoost               | 0.68     | 0.63      | 0.47   | 0.54     | 0.73    |

- CatBoost showed superior classification performance across all metrics.
- Logistic regression offered greater transparency and feature-level interpretability.


## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- CatBoost
