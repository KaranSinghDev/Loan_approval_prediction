Problem Statement
In the financial sector, accurately predicting loan approvals is essential to managing credit risk. Traditional loan evaluation methods, based on applicants’ credit scores and financial profiles, often fail to capture complex relationships in the data, resulting in suboptimal decisions—either high-risk loans are granted or low-risk applicants are denied.

Solution Overview
This project implements a machine learning model using XGBoost to automate and enhance the accuracy of loan approval predictions. Key improvements include:

Feature Engineering: Derived features to capture financial and demographic information that better represents loan applicants.
Class Balancing: Applied SMOTE to address class imbalance, improving model performance for underrepresented classes.
Cross-Validation: Used stratified k-fold cross-validation to ensure the model generalizes well and avoids overfitting.
Compared to traditional and basic machine learning models, this approach provides higher accuracy and reliability, making it suitable for real-world applications in the financial sector.

Dataset
The dataset includes attributes related to demographics, financial status, employment, and credit history, such as:

Personal and Financial Info: person_age, person_income, loan_amnt
Loan and Credit History: loan_grade, loan_int_rate, cb_person_cred_hist_length
Source: The dataset is available at [Dataset Source/URL if applicable].

Model & Hyperparameters
The model is an XGBoost Classifier optimized with the following hyperparameters:

objective: 'binary:logistic'
eval_metric: 'auc'
n_estimators: 500
learning_rate: 0.01
max_depth: 4
min_child_weight: 3
reg_alpha: 0.5, reg_lambda: 1.0
These parameters were selected based on cross-validation to achieve an optimal balance between bias and variance, preventing overfitting while retaining predictive accuracy.

Performance Evaluation
The model’s performance was measured using the AUC-ROC metric:

Cross-Validation AUC-ROC Score: 0.9709
Holdout AUC-ROC Score: 0.9365
