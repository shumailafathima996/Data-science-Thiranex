"""
Data cleaning pipeline for the loan default dataset.
"""
import pandas as pd
import numpy as np

def load_and_clean(path="data/raw_loan_data.csv"):
    df = pd.read_csv(path)
    log = []
    log.append(f"Initial shape: {df.shape}")

    before = len(df)
    df = df.drop_duplicates(subset=["ApplicantID"], keep="first")
    log.append(f"Dropped {before - len(df)} duplicate ApplicantIDs")

    df["CreditScore"] = df["CreditScore"].fillna(df["CreditScore"].median())
    df["AnnualIncome"] = df.groupby("EmploymentType")["AnnualIncome"].transform(
        lambda x: x.fillna(x.median())
    )
    log.append("Imputed missing CreditScore (median) and AnnualIncome (group median by employment type)")

    # cap extreme debt-to-income outliers using IQR
    q1, q3 = df["DebtToIncomeRatio"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 3 * iqr
    n_outliers = (df["DebtToIncomeRatio"] > upper).sum()
    df["DebtToIncomeRatio"] = df["DebtToIncomeRatio"].clip(upper=upper)
    log.append(f"Capped {n_outliers} DebtToIncomeRatio outliers at {upper:.2f}")

    df = df.reset_index(drop=True)
    log.append(f"Final shape: {df.shape}")
    return df, log

if __name__ == "__main__":
    df, log = load_and_clean()
    df.to_csv("data/cleaned_loan_data.csv", index=False)
    with open("data/cleaning_log.txt", "w") as f:
        f.write("\n".join(log))
    print("\n".join(log))
    print("\nMissing values after cleaning:")
    print(df.isna().sum())
