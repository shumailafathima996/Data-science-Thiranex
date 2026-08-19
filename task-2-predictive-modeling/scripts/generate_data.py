"""
Generates a synthetic customer churn dataset for the predictive modeling task.
Target: Churn (1 = customer left, 0 = customer stayed)
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1500

tenure_months = np.random.randint(1, 72, n)
monthly_charges = np.round(np.random.normal(65, 25, n).clip(15, 150), 2)
contract_type = np.random.choice(["Month-to-Month", "One Year", "Two Year"], n, p=[0.55, 0.25, 0.20])
internet_service = np.random.choice(["DSL", "Fiber Optic", "No Internet"], n, p=[0.35, 0.45, 0.20])
tech_support = np.random.choice(["Yes", "No"], n, p=[0.35, 0.65])
online_security = np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])
num_support_calls = np.random.poisson(1.5, n)
paperless_billing = np.random.choice(["Yes", "No"], n, p=[0.6, 0.4])
payment_method = np.random.choice(
    ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"], n
)
senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
total_charges = np.round(monthly_charges * tenure_months * np.random.uniform(0.9, 1.1, n), 2)

churn_logit = (
    -1.5
    + (-0.04 * tenure_months)
    + (0.015 * monthly_charges)
    + (0.35 * (contract_type == "Month-to-Month"))
    + (-0.4 * (contract_type == "Two Year"))
    + (0.3 * (internet_service == "Fiber Optic"))
    + (-0.35 * (tech_support == "Yes"))
    + (-0.3 * (online_security == "Yes"))
    + (0.25 * num_support_calls)
    + (0.2 * (payment_method == "Electronic Check"))
    + (0.3 * senior_citizen)
)
churn_prob = 1 / (1 + np.exp(-churn_logit))
churn = np.random.binomial(1, churn_prob)

df = pd.DataFrame({
    "CustomerID": [f"CUST{2000+i}" for i in range(n)],
    "SeniorCitizen": senior_citizen,
    "TenureMonths": tenure_months,
    "ContractType": contract_type,
    "InternetService": internet_service,
    "TechSupport": tech_support,
    "OnlineSecurity": online_security,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "NumSupportCalls": num_support_calls,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": churn
})

missing_idx = np.random.choice(df.index, size=25, replace=False)
df.loc[missing_idx, "TotalCharges"] = np.nan
dup_idx = np.random.choice(df.index, size=15, replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

df.to_csv("data/raw_customer_data.csv", index=False)
print("Raw dataset created:", df.shape)
print("Churn rate:", df["Churn"].mean().round(3))
