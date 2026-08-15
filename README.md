# PROPSCORE — Propensity-to-Convert Scoring

An end-to-end machine learning pipeline for predicting which users are most likely to convert, using behavioral and demographic features.

## Overview

**PROPSCORE** builds a propensity-to-convert model that assigns each user a probability of conversion. The goal is to identify high-potential users so that marketing and engagement efforts can be prioritized toward the users most likely to convert.

The project uses **gradient-boosted trees**, leakage-safe feature engineering, categorical encoding, and cross-validated evaluation on an imbalanced dataset.

## Key Results

* **600 users** used for model development and evaluation
* **~13% conversion rate**, making conversion prediction an imbalanced classification problem
* **Gradient-boosted tree model** for propensity scoring
* **5-fold cross-validation** for model evaluation
* **AUC-ROC: 0.55**
* Feature importance analysis used to identify the key behavioral drivers of conversion
* The **top 20% of users captured ~21–24% of all converters**
* Achieved approximately **1.1–1.2× lift over random selection**

## Pipeline

```text
Raw User Data
      ↓
Data Preparation
      ↓
Leakage-Safe Feature Engineering
      ↓
Categorical Encoding
      ↓
Gradient-Boosted Model
      ↓
5-Fold Cross-Validation
      ↓
Conversion Probability
      ↓
User Ranking / Propensity Score
      ↓
Top-User Selection
```

## Project Structure

```text
PROPSCORE/
│
├── data_prep.py                  # Data cleaning and feature preparation
├── model.py                      # Model training and evaluation
├── main.py                       # End-to-end pipeline
│
├── data.csv                      # Input user dataset
├── feature_importance.csv        # Ranked feature importance
├── scored_user.csv               # Users with predicted propensity scores
│
└── README.md
```

## Methodology

### 1. Data Preparation

The raw user-level data is processed to prepare it for modeling. This includes:

* Data cleaning
* Feature transformation
* Categorical variable encoding
* Target preparation
* Leakage-safe feature engineering

### 2. Model Training

A **gradient-boosted tree classifier** is trained to estimate the probability that a user will convert.

Because only approximately **13% of users converted**, evaluation focuses on metrics suitable for an imbalanced classification problem rather than accuracy alone.

### 3. Cross-Validation

The model is evaluated using **5-fold cross-validation** to obtain a more robust estimate of out-of-sample performance.

The primary evaluation metric is **AUC-ROC**.

### 4. Propensity Scoring

After training, each user receives a predicted conversion probability.

Users are then ranked from highest to lowest propensity score:

```text
User → Features → Model → Conversion Probability → Rank
```

This ranking can be used to prioritize the highest-potential users for targeted campaigns.

### 5. Feature Importance

Feature importance analysis is used to identify which user behaviors and attributes contribute most to the model's predictions.

The resulting ranked feature-importance dataset is stored in:

```text
feature_importance.csv
```

## Business Insight

The primary objective is not simply to predict conversion, but to **prioritize users efficiently**.

Selecting the top **20% of users based on propensity score** captured approximately **21–24% of all converters**, corresponding to a **1.1–1.2× lift over random selection**.

This provides a framework for targeting high-propensity users while reducing the number of users that need to be contacted.

## Technologies

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Gradient-Boosted Trees**
* **Machine Learning**
* **Feature Engineering**
* **Cross-Validation**
* **Classification & Ranking**

## How to Run

Clone the repository:

```bash
git clone https://github.com/Akarsh-28/PROPSCORE.git
cd PROPSCORE
```

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn
```

Run the pipeline:

```bash
python main.py
```

The pipeline generates scored users and feature-importance results.

## Outputs

### `scored_user.csv`

Contains users ranked according to their predicted propensity to convert.

### `feature_importance.csv`

Contains the model's ranked feature importance, allowing analysis of the behavioral drivers associated with conversion.

## Limitations

The current dataset contains only **600 users**, so model performance may not generalize to a larger population.

Additionally, the current **AUC-ROC of 0.55** indicates relatively limited predictive discrimination. The scoring pipeline should therefore be viewed as an exploratory propensity-ranking framework rather than a production-ready targeting system.

## Future Improvements

* Increase training dataset size
* Perform hyperparameter optimization
* Compare additional models such as XGBoost, LightGBM, and Random Forest
* Optimize probability calibration
* Evaluate Precision@K and Recall@K
* Test the scoring strategy on a future holdout dataset
* Investigate additional behavioral features
* Perform statistical significance testing for lift

## Author

**Akarsh Sinha**

GitHub: [Akarsh-28](https://github.com/Akarsh-28)
