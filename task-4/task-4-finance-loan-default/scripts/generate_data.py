"""
Generates a synthetic loan applicant dataset for a real-world finance
credit risk / loan default prediction project.
Target: LoanDefault (1 = defaulted, 0 = repaid)
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

age = np.random.randint(21, 65, n)
income = np.round(np.random.lognormal(mean=10.8, sigma=0.5, size=n), 2).clip(15000, 300000)
employment_type = np.random.choice(
    ["Salaried", "Self-Employed", "Business Owner", "Unemployed"],
    n, p=[0.55, 0.20, 0.15, 0.10]
)
credit_score = np.round(np.random.normal(650, 90, n)).clip(300, 850).astype(int)
loan_amount = np.round(np.random.lognormal(mean=9.5, sigma=0.6, size=n), 2).clip(1000, 200000)
loan_term_months = np.random.choice([12, 24, 36, 48, 60], n, p=[0.15, 0.25, 0.30, 0.20, 0.10])
num_existing_loans = np.random.poisson(1.2, n).clip(0, 6)
home_ownership = np.random.choice(["Own", "Mortgage", "Rent"], n, p=[0.25, 0.35, 0.40])
marital_status = np.random.choice(["Single", "Married", "Divorced"], n, p=[0.40, 0.45, 0.15])
loan_purpose = np.random.choice(
    ["Debt Consolidation", "Home Improvement", "Business", "Education", "Medical", "Other"],
    n
)

debt_to_income = np.round((loan_amount / (income + 1)) * np.random.uniform(0.8, 1.3, n), 3)

# Build default probability from a realistic combination of risk factors
default_logit = (
    -1.8
    + (-0.012 * (credit_score - 650))
    + (1.8 * debt_to_income)
    + (-0.00003 * income)
    + (0.35 * num_existing_loans)
    + (0.9 * (employment_type == "Unemployed"))
    + (0.35 * (employment_type == "Self-Employed"))
    + (-0.4 * (home_ownership == "Own"))
    + (0.03 * (loan_term_months / 12))
    + (-0.01 * (age - 40) * 0.05)
)
default_prob = 1 / (1 + np.exp(-default_logit))
loan_default = np.random.binomial(1, default_prob)

df = pd.DataFrame({
    "ApplicantID": [f"LN{5000+i}" for i in range(n)],
    "Age": age,
    "AnnualIncome": income,
    "EmploymentType": employment_type,
    "CreditScore": credit_score,
    "LoanAmount": loan_amount,
    "LoanTermMonths": loan_term_months,
    "NumExistingLoans": num_existing_loans,
    "HomeOwnership": home_ownership,
    "MaritalStatus": marital_status,
    "LoanPurpose": loan_purpose,
    "DebtToIncomeRatio": debt_to_income,
    "LoanDefault": loan_default
})

# inject realistic missingness and a few duplicates for a genuine "raw" feel
missing_idx = np.random.choice(df.index, size=40, replace=False)
df.loc[missing_idx, "CreditScore"] = np.nan
missing_idx2 = np.random.choice(df.index, size=25, replace=False)
df.loc[missing_idx2, "AnnualIncome"] = np.nan
dup_idx = np.random.choice(df.index, size=20, replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

df.to_csv("data/raw_loan_data.csv", index=False)
print("Raw dataset created:", df.shape)
print("Default rate:", df["LoanDefault"].mean().round(3))
print("Missing values:\n", df.isna().sum())
