# Credit Card Fraud Detection Using Machine Learning

## Overview

This project focuses on detecting fraudulent credit card transactions using machine learning algorithms. The primary goal is to develop a model that accurately distinguishes between fraudulent and non-fraudulent transactions in a highly imbalanced dataset.

### Problem Statement

Fraudulent credit card transactions pose a significant challenge to financial institutions, leading to substantial financial losses. Detecting these transactions in real time is crucial for preventing fraud and ensuring customer trust.

### Dataset 

The dataset used for this project is sourced from Kaggle and contains credit card transaction data from Europe in September 2013. The dataset consists of 284,807 transactions, of which only 492 (0.172%) are labeled as fraudulent.

Key features include:
- Time: Time elapsed between the transaction and the first transaction in the dataset.
- Amount: The amount of the transaction.
- V1–V28: PCA-transformed features for privacy protection.

The target variable is Class, where:

- 0: Non-fraudulent transactions
- 1: Fraudulent transactions
    
### Solution Approach

Data: A highly imbalanced dataset of credit card transactions, with only 0.17% classified as fraudulent.

#### Data Preprocessing:

1. Load the dataset.
- Check for missing values and handle them appropriately.
- Perform one-hot encoding for categorical variables (if any).
- Normalize or standardize the features, especially for PCA-transformed data.

2. Model Building:
- Split the dataset into training and testing sets.
- Chose Random Forest as the primary model due to its robustness in handling imbalanced datasets and feature importance insights.
- Evaluate the model using appropriate metrics (accuracy, precision, recall, AUPRC).

3. Hyperparameter Tuning:
- Perform Grid Search Cross-Validation to find the best hyperparameters for the Random Forest model.

4. Handling Class Imbalance:
- Since the dataset is highly imbalanced, use techniques like SMOTE (Synthetic Minority Over-sampling Technique) to oversample the minority class and balance the classes.

5. Model Evaluation:
- Evaluate the model using metrics like classification report, ROC-AUC, and Precision-Recall AUC (AUPRC).
- Analyze the confusion matrix and other evaluation metrics to understand the model’s performance.

### Results

- Achieved a recall rate of 100%, ensuring most fraudulent transactions were identified.
- Improved model performance by fine-tuning the classification threshold to balance precision and recall.

### Key Insights

1. Model Performance:
- ROC-AUC Score: 1.000
- AUPRC: 1.000
- Classification report metrics showed perfect scores for precision, recall, and F1-score, reflecting the model's effectiveness in detecting fraudulent transactions.

2. Hyperparameter Tuning:
The best parameters identified were:
- n_estimators: 100
- max_depth: None
- min_samples_split: 2
- class_weight: None

3. SMOTE Effectiveness:
Balanced class distribution achieved after applying SMOTE:
- Fraudulent transactions: 284,315
- Non-fraudulent transactions: 284,315
 
### Future Directions

- Incorporate real-time data streaming and anomaly detection techniques for faster fraud identification.
- Use advanced models like Graph Neural Networks to capture relationships between transactions.

### Source

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
