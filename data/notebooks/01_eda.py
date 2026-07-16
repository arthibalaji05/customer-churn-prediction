import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

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



#day 3 : pie chart visualization

print("\nGenerating charts...")
churn_counts = df['Churn'].value_counts()

plt.figure(figsize=(6,6))
plt.pie(churn_counts, labels=['Stayed','Left'], autopct='%1.1f%%', startangle=90)
plt.title('Customer churn breakdown ie. Stayed vs left')

plt.savefig('churn_piechart.png')
plt.close()
print("      .Saved: churn_pie.png")



# 2. Create the Monthly Charges Distribution Plot
plt.figure(figsize=(10, 5))

# Draw a smooth curve for the customers who stayed (Churn == 'No')
sns.kdeplot(data=df[df['Churn'] == 'No'], x='MonthlyCharges', fill=True, label='Stayed', color='blue', alpha=0.5)

# Draw a smooth curve for the customers who left (Churn == 'Yes')
sns.kdeplot(data=df[df['Churn'] == 'Yes'], x='MonthlyCharges', fill=True, label='Left', color='orange', alpha=0.5)

# Add titles, labels, and a legend
plt.title('Distribution of Monthly Charges by Customer Churn Status')
plt.xlabel('Monthly Charges ($)')
plt.ylabel('Density')
plt.legend()

# Save the second chart as an image
plt.savefig('charges_distribution.png')
plt.close()
print("   • Saved: charges_distribution.png")