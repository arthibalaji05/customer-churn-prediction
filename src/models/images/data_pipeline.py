import pandas as pd
import os

def load_and_clean_data():
    file_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(f"Loading data from: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # --- NEW CLEANING STEPS ---
    
    # 1. Convert TotalCharges to numbers (errors='coerce' turns blank spaces into NaN/Null values)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # 2. Fill those new NaN values with 0 (since they represent brand new customers)
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # 3. Drop the customerID column because it has no predictive power
    df = df.drop(columns=['customerID'])
    
    print("Data loaded and cleaned successfully!")
    
    return df

if __name__ == "__main__":
    df = load_and_clean_data()
    
    print("\nUpdated Data Summary:")
    print(df.info())