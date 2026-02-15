# Forest Fire Weather Index (FWI) Prediction App

A simple Flask web application that predicts FWI using Ridge Regression.

## Features
- Web form to enter 9 input features
- Scales input data using StandardScaler
- Predicts FWI using ridge.pkl

## PROJECT STRUCTURE

|   application.py
|   README.md
|   text.txt
|   
+---dataset
|       Algerian_forest_fires_cleaned_dataset.csv
|       
+---models
|       ridge.pkl
|       scaler.pkl
|       
+---notebooks
|       model_training_Ridg_Lasso_ElasticNet.ipynb
|       Ridg_Lasso_ElasticNet.ipynb
|       
\---templates
        home.html
        index.html
