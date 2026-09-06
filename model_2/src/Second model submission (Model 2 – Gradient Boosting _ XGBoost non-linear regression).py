#!/usr/bin/env python
# coding: utf-8

# Step 1 — Import Libraries for Model 2

# In[1]:


# Step 1: Imports for Model 2
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder

# Optional: XGBoost if installed
# from xgboost import XGBRegressor


# Step 2 – Load Kaggle datasets

# In[2]:


# Step 2: Load Kaggle data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

train.head()


# Step 3 — Encode Categorical Variables

# In[3]:


# Step 3: Encode categorical features BEFORE splitting
cat_cols = train.select_dtypes(include=["object"]).columns

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le


# Step 4 — Recreate Features and Target

# In[4]:


# Step 4: Recreate features and target for Model 2
target_col = "Rings"
X = train.drop(columns=[target_col])
y = train[target_col]

X_test = test.copy()


# Step 5 — Create Train/Validation Split

# In[5]:


# Step 5: New train/validation split for Model 2
X_train2, X_valid2, y_train2, y_valid2 = train_test_split(
    X, y, test_size=0.2, random_state=123
)


# Step 6 — Fit Non‑Linear Model 2 (Gradient Boosting)

# In[6]:


# Step 6: Gradient Boosting model (non-linear)
gbr_model = GradientBoostingRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.9,
    random_state=123
)

gbr_model.fit(X_train2, y_train2)

# Validation predictions + RMSLE
y_valid2_pred = gbr_model.predict(X_valid2)
rmsle_gbr = np.sqrt(mean_squared_log_error(y_valid2, y_valid2_pred))
print("Model 2 - Gradient Boosting RMSLE (validation):", rmsle_gbr)


# Step 7 — Retrain the Chosen Model on Full Training Data

# In[7]:


# Step 7: Retrain Gradient Boosting on full data
gbr_model_full = GradientBoostingRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.9,
    random_state=123
)

gbr_model_full.fit(X, y)

# Predict on Kaggle test set
test_pred_gbr = gbr_model_full.predict(X_test)


# Step 8 — Build Second Submission File

# In[8]:


# Step 8: Create second submission DataFrame
submission_gbr = sample_submission.copy()
submission_gbr["Rings"] = test_pred_gbr

# Save CSV for Kaggle
submission_gbr.to_csv("submission_model2_gradient_boosting.csv", index=False)

submission_gbr.head()

