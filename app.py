import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="Telecom Churn AI",
    page_icon="⚡",
    layout="wide"
)

# 2. Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Trained Model
@st.cache_resource
def load_model():
    return joblib.load("best_churn_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# Get model feature names
model_features = list(getattr(model, 'feature_names_in_', []))

# App Header
st.title("⚡ Telecom Customer Churn Risk Intelligence")
st.write("Predict churn risk for single customers or bulk batch datasets.")

# Create Navigation Tabs
tab1, tab2 = st.tabs(["👤 Single Customer Prediction", "📁 Batch CSV Prediction"])

# ==========================================
# TAB 1: SINGLE CUSTOMER PREDICTION
# ==========================================
with tab1:
    st.subheader("Interactive Profile Assessment")
    
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        st.markdown("### Customer Details")
        tenure = st.slider("Tenure (Months)", 1, 72, 12, key="single_tenure")
        monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0, key="single_monthly")
        total_charges = st.number_input("Total Charges ($)", 18.0, 8500.0, float(tenure * monthly_charges), key="single_total")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="single_contract")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="single_internet")
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ], key="single_payment")
        
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], key="single_senior")
        partner = st.selectbox("Has Partner", ["No", "Yes"], key="single_partner")
        dependents = st.selectbox("Has Dependents", ["No", "Yes"], key="single_dep")
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], key="single_paperless")

    # Feature Vector
    input_data = {
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No': 1 if internet_service == "No" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
    }

    input_df = pd.DataFrame([input_data])

    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_features]

    churn_proba = model.predict_proba(input_df)[0][1]
    churn_percent = round(churn_proba * 100, 1)

    with col_b:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### Risk Gauge")
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_percent,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ef4444" if churn_percent >= 50 else "#3b82f6"},
                'steps': [
                    {'range': [0, 35], 'color': 'rgba(59, 130, 246, 0.2)'},
                    {'range': [35, 65], 'color': 'rgba(234, 179, 8, 0.2)'},
                    {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                ],
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if churn_percent >= 65:
            st.error("🚨 **High Churn Risk!** Immediate retention offer recommended.")
        elif churn_percent >= 35:
            st.warning("⚠️ **Moderate Risk.** Monitor activity.")
        else:
            st.success("🟢 **Low Risk.** Customer is likely to stay.")
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# TAB 2: BATCH CSV PREDICTION
# ==========================================
with tab2:
    st.subheader("Bulk Batch Predictions")
    st.write("Upload a CSV file containing customer data to predict churn risk for multiple accounts simultaneously.")
    
    uploaded_file = st.file_uploader("Upload Customer Dataset (.csv)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write("### Data Preview", batch_data.head())
            
            if st.button("Run Batch Prediction", type="primary"):
                df_proc = batch_data.copy()
                
                # Convert TotalCharges to numeric handling whitespace/nulls
                if 'TotalCharges' in df_proc.columns:
                    df_proc['TotalCharges'] = pd.to_numeric(df_proc['TotalCharges'], errors='coerce').fillna(0)
                
                # Encode binary text columns
                binary_cols = ['Partner', 'Dependents', 'PaperlessBilling', 'PhoneService', 'SeniorCitizen']
                for col in binary_cols:
                    if col in df_proc.columns:
                        df_proc[col] = df_proc[col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0)
                
                # One-hot encoding for categorical columns
                cat_cols = ['Contract', 'InternetService', 'PaymentMethod', 'MultipleLines', 'OnlineSecurity', 
                            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
                
                present_cat_cols = [c for c in cat_cols if c in df_proc.columns]
                if present_cat_cols:
                    df_proc = pd.get_dummies(df_proc, columns=present_cat_cols, drop_first=False)
                
                # Ensure all features expected by the trained model are present
                for col in model_features:
                    if col not in df_proc.columns:
                        df_proc[col] = 0
                
                # Select features in the exact order required by the model
                X_batch = df_proc[model_features].astype(float)
                
                # Model Inference
                probs = model.predict_proba(X_batch)[:, 1]
                preds = (probs >= 0.5).astype(int)
                
                # Append prediction outputs to original dataset
                batch_data['Churn_Probability_%'] = np.round(probs * 100, 1)
                batch_data['Predicted_Churn'] = np.where(preds == 1, 'Yes (High Risk)', 'No (Low Risk)')
                
                st.success(f"✅ Successfully processed {len(batch_data)} customer records!")
                
                # Summary KPIs
                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    high_risk_count = int((preds == 1).sum())
                    st.metric("Total High Risk Customers", f"{high_risk_count} / {len(batch_data)}")
                with col_res2:
                    avg_risk = batch_data['Churn_Probability_%'].mean()
                    st.metric("Average Churn Probability", f"{avg_risk:.1f}%")
                with col_res3:
                    pct_high_risk = round((high_risk_count / len(batch_data)) * 100, 1)
                    st.metric("High Risk Share", f"{pct_high_risk}%")
                
                # Results Table
                st.dataframe(batch_data)
                
                # Export Download Button
                csv_bytes = batch_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Churn Predictions CSV",
                    data=csv_bytes,
                    file_name="telecom_churn_batch_predictions.csv",
                    mime="text/csv"
                )
        except Exception as err:
            st.error(f"Error processing CSV file: {err}")