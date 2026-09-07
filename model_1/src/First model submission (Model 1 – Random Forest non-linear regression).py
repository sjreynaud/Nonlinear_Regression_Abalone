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


# In[3]:


# 3. EDA + Figures Distribution Plots / Warning

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os
import matplotlib.pyplot as plt
import seaborn as sns

# 3. EDA + Figures Distribution Plots
numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

# Automatically creates the "figures" folder if it doesn't exist yet
os.makedirs("NU/DDS8555/Build_Non-Linear_Models_Part 2/figures", exist_ok=True)


plt.figure(figsize=(12, 10))
for i, col in enumerate(numeric_cols):
    plt.subplot(3, 3, i + 1)
    sns.histplot(train[col], kde=True)
    plt.title(f"Distribution of {col}")

plt.tight_layout()
plt.savefig("NU/DDS8555/Build_Non-Linear_Models_Part 2/figures/model1_eda_distributions.png")
plt.show()


# Step 4 — Basic EDA + Missing Data Check

# In[4]:


# Step 4: Quick EDA
print(train.info())
print(train.describe())

print("Missing values in train:")
print(train.isna().sum())

print("Missing values in test:")
print(test.isna().sum())


# Step 5 — Encode Categorical Variables

# In[5]:


#5 fit the encoder on the combined unique values of both the train and test sets. 
# This ensures the encoder learns every possible category beforehand

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 5: Encode categorical features BEFORE splitting
cat_cols = train.select_dtypes(include=["object"]).columns

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    
    if col in test.columns:
        # FIX: Combine unique values from both train and test so 'Always' is recognized
        combined_categories = pd.concat([train[col], test[col]]).dropna().unique()
        le.fit(combined_categories)
        
        # Transform both datasets safely
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
    else:
        # For columns ONLY in train (like your target variable 'NObeyesdad')
        train[col] = le.fit_transform(train[col])
        
    label_encoders[col] = le


# Step 6 — Create Features, Target, and Validation Split

# In[6]:


# Step 6: Encode categorical features BEFORE splitting
cat_cols = train.select_dtypes(include=["object"]).columns

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le


# Step 7 — Fit Non‑Linear Model 1 (Random Forest)

# In[7]:


# Step 7: Define features and target

target_col = "NObeyesdad"

X = train.drop(columns=[target_col])
y = train[target_col]

X_test = test.copy()

# Train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[8]:


# Step 8: Random Forest model
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



# Step 9 — Retrain Random Forest on Full Training Data

# In[9]:


# Step 9: Retrain on full training data
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


# Step 10 — Build First Submission File

# In[10]:


# Step 10: Create first submission DataFrame
submission_rf = sample_submission.copy()
submission_rf["Rings"] = test_pred_rf

# Save CSV for Kaggle
submission_rf.to_csv("NU/DDS8555/Build_Non-Linear_Models_Part 2/submission/submission_model1_random_forest.csv", index=False)

submission_rf.head()


# In[ ]:




