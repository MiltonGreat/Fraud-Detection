# Credit Card Fraud Detection Using Machine Learning

### Overview

This project focuses on detecting fraudulent credit card transactions using machine learning algorithms. The goal is to build a model that can accurately classify transactions as fraudulent or non-fraudulent based on the available dataset. Given that fraud detection is a critical task for financial institutions, this project utilizes a real-world dataset to identify fraudulent patterns, addressing the challenges of class imbalance in the data.

### Dataset 

The dataset used in this project is from Kaggle and contains transactions made by credit card holders in Europe in September 2013. The dataset includes 284,807 transactions, with only 492 (0.172%) being fraudulent. The features consist of transformed data from the original features using Principal Component Analysis (PCA) due to privacy concerns. The two non-PCA transformed features are:

- Time: Time elapsed between the transaction and the first transaction in the dataset.
- Amount: The amount of the transaction.

The target variable is Class, where:

- 0: Non-fraudulent transactions
- 1: Fraudulent transactions
    
### Data Preprocessing:

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

### Evaluation Metrics

Since the dataset is highly imbalanced, accuracy is not a reliable metric. Instead, the model's performance is evaluated using:
- Classification Report: Provides precision, recall, and F1-score for both classes.
- ROC-AUC: Measures the ability of the model to distinguish between classes.
- AUPRC: Measures the area under the Precision-Recall Curve, which is particularly useful for imbalanced datasets.

### Results

The model achieves a high classification performance with an AUPRC score of approximately 1.0, indicating an excellent ability to distinguish between fraudulent and non-fraudulent transactions. The Random Forest Classifier, after hyperparameter tuning and handling class imbalance, performs well even with the highly imbalanced dataset.

### Next Steps

- Model Deployment: Deploy the model as a web service using Flask or FastAPI for real-time fraud detection.
- Real-time Monitoring: Continuously monitor and retrain the model as new transaction data becomes available.
- Alternative Algorithms: Experiment with other algorithms like XGBoost or LightGBM for potentially better performance.

### Source

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
