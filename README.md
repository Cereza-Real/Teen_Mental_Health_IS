# Teen_Mental_Health_IS
BICS 2303: INTELLIGENT SYSTEMS PROJECT (GROUP)

# Teen Mental Health & Depression Predictive Modeling

This project builds an end-to-end machine learning pipeline to predict and analyze depression labels among teenagers based on their social media usage, academic performance, and daily habits. 

To address the severe class imbalance typically found in mental health survey data, the framework integrates **SMOTE (Synthetic Minority Over-sampling Technique)** inside a robust cross-validation loop to ensure unbiased and highly generalizable predictive performance.

---

## Project Architecture & Pipeline

The system is built using an isolated `imblearn` pipeline to safely process incoming data without risking data leakage during training and validation.

[ Raw Input Data: X ]
             │
  ┌──────────┴──────────┐
  ▼                     ▼
┌──────────┐          ┌──────────┐
│Numerical │          │Categorical│
│ Columns  │          │ Columns  │
└─────┬────┘          └─────┬────┘
      │                     │
      ▼                     ▼
┌──────────┐          ┌──────────┐
│  Median  │          │   Most   │  <-- SimpleImputer()
│ Imputer  │          │ Frequent │
└─────┬────┘          └─────┬────┘
      │                     │
      ▼                     ▼
┌──────────┐          ┌──────────┐
│ Standard │          │ One-Hot  │  <-- StandardScaler() / OneHotEncoder()
│  Scaler  │          │ Encoder  │
└─────┬────┘          └─────┬────┘
 │                     │
 └──────────┬──────────┘
            │
            ▼
[ ColumnTransformer Out ] ───► (Dense Encoded Feature Matrix)
            │
            ▼
     ┌─────────────┐
     │    SMOTE    │  <-- Dynamically balances training classes
     └──────┬──────┘      
            │
            ▼
     ┌─────────────┐
     │ Classifier  │  <-- Optimized via GridSearch (RF, GB, or DT)
     └─────────────┘


---

## Dataset Features

The model utilizes demographic, behavioural, and emotional health indicators:

* **Categorical Features:** `gender`, `platform_usage`, `social_interaction_level`
* **Numerical Features:** `age`, `daily_social_media_hours`, `sleep_hours`, `screen_time_before_sleep`, `academic_performance`, `physical_activity`, `stress_level`, `anxiety_level`, `addiction_level`
* **Target Label:** `depression_label` (Binary: Not Depressed / Depressed)

---


### 1. Prerequisites
Ensure you have Python 3.8+ installed. Install the required dependencies using:

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib

library required
pandas
numpy
matplotlib
seaborn
joblib
scikit-learn
imbalanced-learn
