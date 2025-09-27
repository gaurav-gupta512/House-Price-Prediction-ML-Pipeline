# House Price Prediction Project

This repository contains my House Price Prediction project using Python and Scikit-learn.

## Project Description

Built an end-to-end machine learning pipeline to preprocess housing data, train a Random Forest model, and generate automated price predictions.

## Features

- Data preprocessing with missing value imputation and standard scaling for numerical features.
- One-hot encoding for categorical features.
- Stratified sampling based on income categories to maintain distribution in training data.
- Pipeline persistence using Joblib for future inference.
- Automated batch predictions on new data CSV files with results exported to output CSV.

## Tools & Libraries

- Python
- Pandas, NumPy
- Scikit-learn
- Joblib

## How to Use

1. Place your input CSV file as `input.csv`.
2. Run the inference script. If the trained model exists, it will generate predictions and save them to `output.csv`. If not, it will train the model first and then generate predictions.

## File Structure

- `housing.csv` – original dataset for training
- `input.csv` – CSV file with new data for predictions
- `model.pkl` – saved Random Forest model
- `pipeline.pkl` – saved preprocessing pipeline
- `output.csv` – predictions output
