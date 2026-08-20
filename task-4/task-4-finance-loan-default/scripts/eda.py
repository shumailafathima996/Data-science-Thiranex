"""
EDA for the loan default dataset: statistical summaries, correlations, key visuals.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

df = pd.read_csv("data/cleaned_loan_data.csv")

numeric_cols = ["Age", "AnnualIncome", "CreditScore", "LoanAmount",
                 "LoanTermMonths", "NumExistingLoans", "DebtToIncomeRatio"]
corr = df[numeric_cols + ["LoanDefault"]].corr()
corr.to_csv("data/correlation_matrix.csv")
print("Correlation with LoanDefault:")
print(corr["LoanDefault"].sort_values(ascending=False))

# 1. Correlation heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap — Loan Features")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png")
plt.close()

# 2. Default rate by credit score band
df["CreditScoreBand"] = pd.cut(df["CreditScore"], bins=[300, 580, 670, 740, 800, 850],
                                labels=["Poor (<580)", "Fair", "Good", "Very Good", "Excellent"])
plt.figure(figsize=(8, 5))
rate_by_band = df.groupby("CreditScoreBand", observed=True)["LoanDefault"].mean()
sns.barplot(x=rate_by_band.index, y=rate_by_band.values, hue=rate_by_band.index, palette="RdYlGn", legend=False)
plt.title("Default Rate by Credit Score Band")
plt.ylabel("Default Rate")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("images/default_rate_by_credit_band.png")
plt.close()

# 3. Debt-to-income vs default
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="LoanDefault", y="DebtToIncomeRatio", hue="LoanDefault", palette="Set2", legend=False)
plt.xticks([0, 1], ["Repaid", "Defaulted"])
plt.title("Debt-to-Income Ratio by Loan Outcome")
plt.tight_layout()
plt.savefig("images/dti_by_outcome.png")
plt.close()

# 4. Default rate by employment type
plt.figure(figsize=(8, 5))
rate_by_emp = df.groupby("EmploymentType")["LoanDefault"].mean().sort_values(ascending=False)
sns.barplot(x=rate_by_emp.index, y=rate_by_emp.values, hue=rate_by_emp.index, palette="mako", legend=False)
plt.title("Default Rate by Employment Type")
plt.ylabel("Default Rate")
plt.tight_layout()
plt.savefig("images/default_rate_by_employment.png")
plt.close()

# 5. Income distribution by outcome
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x="AnnualIncome", hue="LoanDefault", fill=True, common_norm=False, alpha=0.4, palette="Set1")
plt.title("Annual Income Distribution by Loan Outcome")
plt.xlim(0, 200000)
plt.tight_layout()
plt.savefig("images/income_distribution_by_outcome.png")
plt.close()

print("\nEDA visualizations saved to images/")
