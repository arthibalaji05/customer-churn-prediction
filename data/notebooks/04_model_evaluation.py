import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Load Preprocessed Data
print(" Loading preprocessed dataset...")
df = pd.read_csv("data/telecom_churn_processed.csv")

if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

# Convert any remaining categorical text columns into dummies
for col in df.select_dtypes(include=['object', 'category']).columns:
    if col != 'Churn':
        df = pd.get_dummies(df, columns=[col], drop_first=True, dtype=int)

# Separate features (X) and target variable (y)
X = df.drop(columns=['Churn'])
y = df['Churn']

# 2. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train Model with Class Balancing
print(" Training Optimized Logistic Regression with Class Balancing...")
tuned_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
tuned_model.fit(X_train, y_train)

# 4. Predictions & Evaluation Metrics
y_pred = tuned_model.predict(X_test)
y_proba = tuned_model.predict_proba(X_test)[:, 1]

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", round(roc_auc_score(y_test, y_proba), 4))

# 5. Save the Final Model Artifact
joblib.dump(tuned_model, "best_churn_model.pkl")
print("\n Best model saved successfully as 'best_churn_model.pkl'!")