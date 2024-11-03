import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import numpy as np

# Load the dataset
train_df = pd.read_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Playgraoung\Loan_approval_prediction\train.csv")
test_df = pd.read_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Playgraoung\Loan_approval_prediction\test.csv")

# Separate features and target from the training set
X = train_df.drop(columns=['id', 'loan_status'])  # Drop id and target
y = train_df['loan_status']

# List of columns by type for preprocessing
categorical_onehot = ['person_home_ownership', 'loan_intent']
categorical_ordinal = ['loan_grade']
numerical_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
                  'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

# Define preprocessing for numerical columns (scaling)
numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# Define preprocessing for categorical columns (One-Hot and Ordinal Encoding)
categorical_transformer = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_onehot),
    ('ordinal', OrdinalEncoder(categories=[['G', 'F', 'E', 'D', 'C', 'B', 'A']]), categorical_ordinal)
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_onehot + categorical_ordinal)
])

# Split data into train + public validation and hold-out validation set
X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Preprocess the training and holdout validation sets
X_train_transformed = preprocessor.fit_transform(X_train_full)
X_holdout_transformed = preprocessor.transform(X_holdout)

# Apply SMOTE to the preprocessed training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train_full)

# Define the model
xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=500,
    learning_rate=0.01,
    max_depth=4,
    min_child_weight=3,
    scale_pos_weight=1.0,
    reg_alpha=0.5,
    reg_lambda=1.0,
    use_label_encoder=False
)

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store the AUC-ROC scores
cv_auc_scores = []

# Cross-validation loop
for train_index, val_index in cv.split(X_train_resampled, y_train_resampled):
    X_cv_train, X_cv_val = X_train_resampled[train_index], X_train_resampled[val_index]
    y_cv_train, y_cv_val = y_train_resampled[train_index], y_train_resampled[val_index]
    
    # Train the model without early stopping
    xgb_model.fit(X_cv_train, y_cv_train, verbose=False)

    # Predict probabilities for validation fold
    y_cv_val_pred_proba = xgb_model.predict_proba(X_cv_val)[:, 1]
    
    # Evaluate AUC-ROC score
    cv_auc = roc_auc_score(y_cv_val, y_cv_val_pred_proba)
    cv_auc_scores.append(cv_auc)

# Calculate mean CV AUC-ROC score
mean_cv_auc = np.mean(cv_auc_scores)
print(f"Mean Cross-Validation AUC-ROC Score: {mean_cv_auc:.4f}")

# Validate the final model on the holdout validation set
xgb_model.fit(X_train_resampled, y_train_resampled)  # Train on full resampled training set
y_holdout_pred_proba = xgb_model.predict_proba(X_holdout_transformed)[:, 1]
holdout_auc_score = roc_auc_score(y_holdout, y_holdout_pred_proba)
print(f"Holdout Validation AUC-ROC Score: {holdout_auc_score:.4f}")

# Predict probabilities for the test set
X_test_transformed = preprocessor.transform(test_df.drop(columns=['id']))  # Transform test set
test_pred_proba = xgb_model.predict_proba(X_test_transformed)[:, 1]

# Prepare the submission file
submission = pd.DataFrame({'id': test_df['id'], 'loan_status': test_pred_proba})
submission.to_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Playgraoung\Loan_approval_prediction\sample_submission.csv", index=False)
print("Submission file 'loan_approval_predictions.csv' is ready.")
