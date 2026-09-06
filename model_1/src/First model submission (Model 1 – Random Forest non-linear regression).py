#!/usr/bin/env python
# coding: utf-8

# First model submission (Model 1 – Random Forest non-linear regression)

# Step 1 – Import Libraries

# In[1]:


# Step 1: Imports for Model 1
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder


# Step 2 – Load Kaggle datasets

# In[2]:


# Step 2: Load Kaggle data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

train.head()


# Step 3 — Basic EDA + Missing Data Check

# In[3]:


# Step 3: Quick EDA
print(train.info())
print(train.describe())

print("Missing values in train:")
print(train.isna().sum())

print("Missing values in test:")
print(test.isna().sum())


# Step 4 — Encode Categorical Variables

# In[4]:


# Step 4: Encode categorical features BEFORE splitting
cat_cols = train.select_dtypes(include=["object"]).columns

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le


# Step 5 — Create Features, Target, and Validation Split

# In[5]:


# Step 5: Define features and target
target_col = "Rings"
X = train.drop(columns=[target_col])
y = train[target_col]

X_test = test.copy()

# Train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 6 — Fit Non‑Linear Model 1 (Random Forest)

# In[6]:


# Step 6: Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Validation predictions + RMSLE
y_valid_pred = rf_model.predict(X_valid)
rmsle_rf = np.sqrt(mean_squared_log_error(y_valid, y_valid_pred))
print("Model 1 - Random Forest RMSLE (validation):", rmsle_rf)



# Step 7 — Retrain Random Forest on Full Training Data

# In[7]:


# Step 7: Retrain on full training data
rf_model_full = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

rf_model_full.fit(X, y)

# Predict on Kaggle test set
test_pred_rf = rf_model_full.predict(X_test)


# Step 8 — Build First Submission File

# In[8]:


# Step 8: Create first submission DataFrame
submission_rf = sample_submission.copy()
submission_rf["Rings"] = test_pred_rf

# Save CSV for Kaggle
submission_rf.to_csv("submission_model1_random_forest.csv", index=False)

submission_rf.head()


# In[ ]:




