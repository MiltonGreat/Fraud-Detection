# Credit Card Fraud Detection Using Machine Learning

## Overview

This project focuses on detecting fraudulent credit card transactions using machine learning algorithms. The goal is to build a model that can accurately classify transactions as fraudulent or non-fraudulent based on the available dataset. 

### Problem Statement

Detecting fraudulent credit card transactions in real-time is essential for financial institutions to prevent significant losses.

### Dataset 

The dataset used in this project is from Kaggle and contains transactions made by credit card holders in Europe in September 2013. The dataset includes 284,807 transactions, with only 492 (0.172%) being fraudulent. The features consist of transformed data from the original features using Principal Component Analysis (PCA) due to privacy concerns. The two non-PCA transformed features are:

- Time: Time elapsed between the transaction and the first transaction in the dataset.
- Amount: The amount of the transaction.

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
- Train a Random Forest Classifier on the training data.
- Evaluate the model using appropriate metrics (accuracy, precision, recall, AUPRC).

3. Hyperparameter Tuning:
- Perform Grid Search Cross-Validation to find the best hyperparameters for the Random Forest model.

4. Handling Class Imbalance:
- Since the dataset is highly imbalanced, use techniques like SMOTE (Synthetic Minority Over-sampling Technique) to oversample the minority class and balance the classes.

5. Model Evaluation:
- Evaluate the model using metrics like classification report, ROC-AUC, and Precision-Recall AUC (AUPRC).
- Analyze the confusion matrix and other evaluation metrics to understand the model’s performance.

### Results

- Achieved a recall rate of 95%, ensuring most fraudulent transactions were identified.
- Improved model performance by fine-tuning the classification threshold to balance precision and recall.

### Key Insights

- Handling imbalanced datasets is critical for fraud detection models.
- A high recall rate minimizes false negatives, which is crucial for detecting fraud effectively.

### Future Directions

- Incorporate real-time data streaming and anomaly detection techniques for faster fraud identification.
- Use advanced models like Graph Neural Networks to capture relationships between transactions.

### Source

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
