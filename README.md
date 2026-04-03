# Advanced Transaction Anomaly Detection System

This repository contains a state-of-the-art anomaly and fraud detection engine. Designed to identify suspicious financial transactions, the system leverages a multi-tiered architecture that spans from traditional human logic to advanced ensemble machine learning. 

The project is cleanly modularized into three distinct modeling approaches:

---

## 🏗️ 1. Rule-Based Engine (`models/rule_based/fi.py`)

The foundational layer of the system uses deterministic, hardcoded rules designed to mimic the investigative logic of a human fraud analyst. It calculates baseline behaviors for each user and flags real-time deviations.

It evaluates transactions across 6 primary vectors:
1. **Amount Anomaly**: Transactions statistically larger than the user’s personal baseline mean (using Z-scores).
2. **Temporal Deviation**: Activity during hours where the user historically rarely transacts (e.g., 3:00 AM).
3. **Location Deviation**: Physical transactions occurring thousands of miles from the user's calculated geographical center.
4. **Velocity Check**: Impossible high-frequency transaction bursts (e.g., 5+ swipes in under 60 minutes).
5. **Category Deviation**: Spending in merchant categories the user has never utilized before.
6. **Geographic Impossibility**: Consecutive swipes at locations physically impossible to travel between in the elapsed time (i.e., requiring travel speeds > 900 km/h).

*Transactions flagged by 2 or more rules receive a high **Composite Fraud Score**.*

---

## 🤖 2. Supervised Machine Learning (`models/supervised_xgboost/xgboost_anomaly_detection.py`)

The second layer replaces raw human logic with a highly accurate **XGBoost Classifier**. 

* **Feature Engineering**: It auto-generates 12+ complex mathematical features from the raw dataset (e.g., Haversine distances, time-since-last-transaction, relative amount ratios).
* **Supervised Training**: The algorithm trains on billions of historical patterns to explicitly map input combinations to known fraudulent outcomes (or uses the Rule-Based composite scores as synthetic labels).
* **Output**: Rather than a binary Yes/No, it provides a nuanced **Fraud Probability Score** (0% to 100%), allowing businesses to set their own risk thresholds. 

---

## 🧠 3. Generic Unsupervised Ensemble (`models/unsupervised_generic/generic_anomaly_detection.py`)

The final layer is a completely agnostic, mathematically driven pipeline. It does not require labeled training data, nor does it require manual column mapping. You can upload **any arbitrary numerical CSV**, and it will run autonomously.

It creates a "Wisdom of the Crowds" ensemble using three distinct mathematical architectures:
1. **Isolation Forest**: Isolates anomalies by partitioning high-dimensional feature spaces using random decision trees.
2. **DBSCAN (with PCA)**: Compresses the data to a dense mathematical space using Principal Component Analysis, then calculates a dynamic radius (`eps`) to group normal behaviors into clusters. Anything falling in the sparse void outside these clusters is flagged.
3. **Local Outlier Factor (LOF)**: Evaluates the specific density of a transaction relative to its immediate neighbors to catch subtle, localized deviations.

*The models hold a **democratic vote**. If a transaction receives 2 out of 3 votes, it is flagged as a high-confidence outlier.*

---

## 🗂️ Project Structure

```text
├── data/                    # Reserved for raw input CSVs
├── results/                 # All generated outputs, CSVs, and visualization plots
└── models/
    ├── rule_based/          # Core Logic engine + Jupyter Notebooks
    ├── supervised_xgboost/  # XGBoost Predictive code + Notebooks
    └── unsupervised_generic/# Unsupervised CSV Pipeline + Notebooks
```

### Usage
Each tier functions entirely independently. You can run any of the models by navigating to their respective folder and executing the python script. 

`python models/unsupervised_generic/generic_anomaly_detection.py`

*(Every model script will automatically deposit output datasets and `.png` data visualizations backwards into the master `results/` folder for review).*
