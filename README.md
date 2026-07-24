# Telecom Customer Churn Risk Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning web application designed to identify and predict customer churn risk for telecommunication providers. Built with **Scikit-Learn**, **Streamlit**, and **Plotly**, this platform features both real-time interactive single-profile risk assessment and high-speed bulk batch CSV predictions.

---

## Project Overview

Customer churn is a critical business metric for subscription-based telecom providers. Identifying churn drivers before a customer leaves allows companies to take proactive retention measures. 

This project delivers a complete data science pipeline—from exploratory data analysis (EDA) and data preprocessing to model training, evaluation, and production-ready Streamlit dashboard deployment.

---

## Core Features

- *Interactive Single Customer Profiler:** Real-time risk gauge scoring with dynamic inputs (contract type, tenure, monthly charges, internet service, payment methods).
- *High-Volume Batch CSV Inference:** Upload raw customer datasets (e.g., 7,000+ accounts) for automated feature encoding, bulk churn probability scoring, and statistical KPI summaries.
- **Interactive Visualizations:** Sleek risk gauges, percentage metrics, and probability distributions powered by Plotly.
- **One-Click Export:** Download processed batch predictions directly as an updated `.csv` report containing predicted risk labels and probabilities.

---

#Tech Stack & Libraries

- **Language:** Python
- **Machine Learning & Analytics:** Scikit-Learn, Pandas, NumPy, Joblib
- **Interactive UI & Dashboard:** Streamlit, Custom CSS
- **Data Visualization:** Plotly Express & Plotly Graph Objects
- **Version Control:** Git, GitHub

---

#Getting Started Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/arthibalaji05/customer-churn-prediction.git](https://github.com/arthibalaji05/customer-churn-prediction.git)
cd customer-churn-prediction
