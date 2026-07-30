
# Project Overview

## Project Overview
This project uses supervised machine learning to classify **lithology** from well log data using the **XGBoost** algorithm. The objective is to accurately predict lithology classes based on petrophysical well log measurements.

## Results
- **Accuracy:** 0.93
- **F1-Score:** 0.86

## Workflow

### 1. the Problem 
Lithology interpretation can be subjective and time-consuming. This project aims to assist engineers by providing fast and consistent lithology predictions using machine learning, helping reduce interpretation variability rather than replacing expert judgment.

---

### 2. Quick Look at the Data
Perform an initial exploration of the dataset using basic pandas methods:

- `info()`
- `describe()`

---

### 3. Plot a Histogram for Each Feature
Visualize the distribution of each feature to better understand the data and identify potential outliers.

---

### 4. Split the Data into Training and Testing Sets
Split the dataset into training and testing sets. All preprocessing, feature engineering, and model development are performed **only on the training data** to prevent data leakage. The split is based on **well** and **depth** to ensure a realistic evaluation.

---

### 5. Visualize the Data in Depth
Perform detailed exploratory data analysis (EDA) to identify patterns, correlations, missing values, and outliers.

---

### 6. Feature Engineering
Create and transform features that better represent the underlying geological information and improve model performance.

---

### 7. Data Cleaning & Preprocessing
- Remove features with more than **70% missing values**.
- Clean missing values from the core logging features.
- Prepare the dataset for model training.

---

### 8. Build the Model
Train an **XGBoost** classifier for lithology prediction. XGBoost was selected because it provides excellent performance on structured tabular data, handles complex feature interactions, and is robust against overfitting.

---

### 9. Cross-Validation and Evaluation
Evaluate the model's stability and generalization performance using cross-validation. The cross-validation results are available in the corresponding output file.

---

### 10. Final Test on the Test Set
Evaluate the final model on the previously unseen test dataset. The final evaluation results are available in the corresponding output file.