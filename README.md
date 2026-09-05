
**Nonlinear Regression with the Abalone Dataset**

**Overview**

This repository contains all work completed for Assignment 4 in the Predictive Analysis course at National University. The project focuses on building and evaluating **nonlinear regression models** to predict abalone age (measured in rings) using the Kaggle Regression with Abalone Dataset. The repository includes full Exploratory Data Analysis (EDA), preprocessing, model development, Kaggle submissions, and an APA‑formatted written report addressing required ISLR Python questions.

**Project Objectives**
  - Complete **Conceptual Question #3** and **Applied Question #8** from ISLR Python
  - Build two nonlinear regression models using methods discussed in lecture
  - Submit both models to Kaggle and provide evidence of submission
  - Interpret model performance and assumptions
  - Provide an APA‑formatted report with scholarly references
  - Publish all code and deliverables in a reproducible GitHub repository

**Repository Contents**

  - **/notebooks/** — Jupyter notebooks for both models
      **- Model 1 – Random Forest Regression**
      **- Model 2 – Gradient Boosting Regression (XGBoost optional)**
  - **/submissions/** — Kaggle submission CSV files
  - **/report/** — APA‑formatted written report
  - **/eda/** — EDA outputs and preprocessing steps
  - **/references/** — Scholarly sources used in the assignment
  - **README.md** — Project documentation

**Dataset**

The Abalone dataset contains physical measurements such as length, diameter, height, and various weight metrics. The target variable is Rings, which approximates the age of the abalone.
The dataset includes:

  - 90,615 training rows
  - 10 numerical predictors
  - 1 categorical predictor (Sex)
  - No missing values

**Methods**

Two nonlinear regression models were developed:

**1. Random Forest Regression**

  - 400 estimators
  - Full-depth trees
  - RMSLE (validation): 0.1549
  - Submitted to Kaggle as submission_model1_random_forest.csv

**2. Gradient Boosting Regression**

  - 600 estimators
  - Learning rate: 0.05
  - Subsample: 0.9
  - RMSLE (validation): 0.1494
  - Submitted to Kaggle as submission_model2_gradient_boosting.csv

**Model Interpretation**

Both models captured nonlinear relationships between abalone physical measurements and age. Gradient Boosting achieved the best performance due to its sequential error‑correcting structure. Feature importance analysis (not included in the PDF output) typically highlights length, diameter, and whole weight as dominant predictors.

**Assumptions**

Tree‑based nonlinear models assume:

    - Independent observations
    - Representative training data
    - Consistent measurement scales
      They do not require linearity, homoscedasticity, or normality, 
      making them robust for biological datasets like abalone measurements.

**Kaggle Submission Evidence**

Both models were successfully submitted to the Kaggle competition. Submission files and screenshots are included in the /submissions/ folder.

**References**

Three scholarly sources were used to support:

  - Nonlinear modeling theory
  - EDA principles
  - Missing data handling

**How to Use This Repository**

Clone the repository and run the notebooks in the /notebooks/ folder.
All code is written in Python using scikit‑learn and follows reproducible machine‑learning practices.
