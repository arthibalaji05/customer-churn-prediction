import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. Load Preprocessed Data
DATA_PATH = "data/telecom_churn_processed.csv"
print("Loading preprocessed dataset...")
df = pd.read_csv(DATA_PATH)

# Drop identifier columns if present
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

# Convert any remaining text/object columns to numbers automatically
for col in df.select_dtypes(include=['object', 'category']).columns:
    if col != 'Churn':
        # Apply dummy/one-hot encoding for remaining string columns
        df = pd.get_dummies(df, columns=[col], drop_first=True, dtype=int)

# Separate features (X) and target variable (y)
X = df.drop(columns=['Churn'])
y = df['Churn']

# Ensure target variable y is numeric (0/1)
if y.dtype == 'object':
    y = y.map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0}).fillna(y)

# 2. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f" Data split complete! Training samples: {len(X_train)}, Testing samples: {len(X_test)}\n")

# 3. Define Models to Evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
}

# 4. Train and Evaluate
results = []

print("Training models and calculating evaluation metrics...\n")
for name, model in models.items():
    # Fit model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate performance metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    
    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(roc, 4)
    })

# 5. Display Comparison Matrix
results_df = pd.DataFrame(results)
print("--- MODEL COMPARISON MATRIX ---")
print(results_df.to_string(index=False))