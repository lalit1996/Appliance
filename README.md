📌 Overview

This repository explores appliance energy consumption prediction using Linear Regression with time-aware cross-validation.
The goal is to understand where linear models work and where they fail when applied to real-world energy data.

Key takeaway: Linear Regression is not always perfect, especially for behavioral and time-dependent data.

📊 Dataset

Source: UCI Machine Learning Repository

Name: Appliances Energy Prediction

Frequency: Every 10 minutes

Targets:

Appliances — energy consumption (Wh)

lights — lighting energy consumption (Wh)

⚙️ Feature Engineering

Extracted time-based features:

Hour, Minute, Day, Month

Aggregated indoor temperature and humidity values

Removed unnecessary columns

Ensured chronological ordering (no shuffling)

🧠 Model

Linear Regression

Implemented using scikit-learn Pipeline:

StandardScaler → LinearRegression


✅ Feature scaling applied only to X
✅ Target (y) not scaled

🔁 Validation Strategy

TimeSeriesSplit (5 folds)

Ensures:

Training on past data only

Testing on future data

No data leakage

📈 Evaluation Metric

R² Score (Coefficient of Determination)

Observed R² scores across folds:

-0.18, 0.12, 0.03, -0.06, 0.05
