import pandas as pd
import os

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(f"Dataset loaded successfully! Shape: {df.shape}")


print("checking for hidden missing values")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors= 'coerce')
print(f"True missing values: {df['TotalCharges'].isnull().sum()}")



# 2. Fill the missing values with 0
df['TotalCharges'] = df['TotalCharges'].fillna(0)
print(f"Missing values after filling: {df['TotalCharges'].isnull().sum()}")


#step 2 binary encoding as systems cant understand yes or no using map()

print("encoding binary columns.....")

binary_map = {'Yes':1, 'No':0}

df['Churn'] = df['Churn'].map(binary_map)
df['Partner'] = df['Partner'].map(binary_map)
df['Dependents'] = df['Dependents'].map(binary_map)



print(f"Unique values in Churn column now: {df['Churn'].unique()}")
print(f"Unique values iN Partner column now: {df['Partner'].unique()}")

#since some columns have more than 2 values
print("performing one hot encoding........")
categorical_cols = ['Contract', 'PaymentMethod', 'InternetService']

df = pd.get_dummies(data=df, columns=categorical_cols, drop_first=True, dtype=int)
print("A few of our new engineered columns:")
print([col for col in df.columns if 'Contract' in col or 'InternetService' in col])


# 5. Save the preprocessed dataset
output_path = "data/telecom_churn_processed.csv"
df.to_csv(output_path, index=False)
print(f"🎉 Preprocessing complete! Clean dataset saved to: {output_path}")