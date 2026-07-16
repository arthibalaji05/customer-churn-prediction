import pandas as pd
import os

file_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = pd.read_csv(file_path)


#quick cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce').fillna(0)


#This part of the code is designed to answer one crucial business question: "What percentage of our total customer base is actually leaving us?"

# Question 1: How many people are leaving?
churn_counts = df['Churn'].value_counts(normalize=True) * 100
print(f"   • Customers who STAYED: {churn_counts['No']:.2f}%")
print(f"   • Customers who LEFT:   {churn_counts['Yes']:.2f}%")


# Question 2: Is it about the money?
avg_monthly = df.groupby('Churn')['MonthlyCharges'].mean()
print(f"   • Stayed Customers pay: ${avg_monthly['No']:.2f} / month")
print(f"   • Churned Customers paid: ${avg_monthly['Yes']:.2f} / month")

