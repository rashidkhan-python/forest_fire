# Forest Fire Weather Index (FWI) Prediction App

A Flask web application that predicts **Fire Weather Index (FWI)** using a trained **Ridge Regression** model.

## Overview

This project takes 9 weather/environment input features from a web form, scales them using a saved `StandardScaler`, and predicts the FWI value using a saved Ridge model.

## Features

- User-friendly web form for input
- Prediction using trained ML model (`ridge.pkl`)
- Input scaling using `scaler.pkl`
- Result displayed directly on the web page

## Tech Stack

- Python
- Flask
- scikit-learn
- NumPy
- Pandas
- Pickle

## Project Structure

```text
.
├── application.py
├── README.md
├── text.txt
├── dataset/
│   └── Algerian_forest_fires_cleaned_dataset.csv
├── models/
│   ├── ridge.pkl
│   └── scaler.pkl
├── notebooks/
│   ├── model_training_Ridg_Lasso_ElasticNet.ipynb
│   └── Ridg_Lasso_ElasticNet.ipynb
└── templates/
    ├── home.html
    └── index.html
