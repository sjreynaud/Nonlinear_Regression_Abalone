#!/usr/bin/env python
# coding: utf-8

# Step 1 — Import Libraries for Model 2

# In[1]:


# Step 1: Imports for Model 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder

sns.set(style="whitegrid")


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

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 3: Encode categorical features BEFORE splitting
cat_cols = train.select_dtypes(include=["object"]).columns

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    
    # Check if the column exists in the test set
    if col in test.columns:
        # FIX: Combine train and test unique values so 'Always' is learned
        combined_categories = pd.concat([train[col], test[col]]).dropna().unique()
        le.fit(combined_categories)
        
        # Safely transform both datasets
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
    else:
        # Handle columns only in train (like your target 'NObeyesdad')
        train[col] = le.fit_transform(train[col])
        
    label_encoders[col] = le


# Step 4 — Recreate Features + Split and Target

# In[4]:


# # Step 4: Recreate features and target for Model 2

target_col = "NObeyesdad"

X = train.drop(columns=[target_col])
y = train[target_col]
X_test = test.copy()

X_train2, X_valid2, y_train2, y_valid2 = train_test_split(
    X, y, test_size=0.2, random_state=123
)


# Step 5 — Create Train/Validation Split

# In[5]:


# Step 5: New train/validation split for Model 2
X_train2, X_valid2, y_train2, y_valid2 = train_test_split(
    X, y, test_size=0.2, random_state=123
)


# Step 6 — Fit Non‑Linear Model 2 (Train Gradient Boosting)

# In[6]:


# Step 6: Train Gradient Boosting model (non-linear classification)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Switched from Regressor to Classifier
gbc_model = GradientBoostingClassifier(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.9,
    random_state=123
)

# Train the classification model
gbc_model.fit(X_train2, y_train2)

# # Validation predictions + Accuracy
y_valid2_pred = gbc_model.predict(X_valid2)

# Switched metric from RMSLE to Accuracy Score (standard for classification)
accuracy_gbc = accuracy_score(y_valid2, y_valid2_pred)

print("Model 2 - Gradient Boosting Accuracy (validation):", accuracy_gbc)


# 7. GBR Figures Feature Importance

# In[7]:


# 7. GBC Figures Feature Importance

importances = gbc_model.feature_importances_
feat_names = X_train2.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feat_names)
plt.title("Gradient Boosting Feature Importance")
plt.savefig("NU/DDS8555/Build_Non-Linear_Models_Part 2/figures/model2_gbr_feature_importance.png")
plt.show()


# In[8]:


# 8 Actual vs Predicted

plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_valid2, y=y_valid2_pred, alpha=0.3)
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.title("GBR: Actual vs Predicted")
plt.savefig("NU/DDS8555/Build_Non-Linear_Models_Part 2/figures/model2_gbr_actual_vs_pred.png")
plt.show()


# In[9]:


# 9. Residuals

residuals_gbr = y_valid2 - y_valid2_pred

plt.figure(figsize=(6, 4))
sns.histplot(residuals_gbr, kde=True)
plt.title("GBR Residual Distribution")
plt.savefig("NU/DDS8555/Build_Non-Linear_Models_Part 2/figures/model2_gbr_residuals.png")
plt.show()


# Step 10 — Retrain the Chosen Model on Full Training Data

# In[10]:


# 10. Retrain Gradient Boosting on full data

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


# In[11]:


#11. GBR Prediction Distribution

plt.figure(figsize=(6, 4))
sns.histplot(test_pred_gbr, kde=True)
plt.title("GBR Test Predictions Distribution")
plt.savefig("NU/DDS8555/Build_Non-Linear_Models_Part 2/figures/model2_gbr_prediction_distribution.png")
plt.show()


# Step 8 — Build Second Submission File

# In[12]:


# Step 8: Create second submission DataFrame
submission_gbr = sample_submission.copy()
submission_gbr["Rings"] = test_pred_gbr

# Save CSV for Kaggle
submission_gbr.to_csv("NU/DDS8555/Build_Non-Linear_Models_Part 2/submission/submission_model2_gradient_boosting.csv", index=False)

submission_gbr.head()


# In[ ]:




