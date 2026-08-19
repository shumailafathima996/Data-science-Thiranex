"""
Data Cleaning Pipeline
Handles: missing values, duplicates, inconsistent formatting, outliers, wrong dtypes.
"""
import pandas as pd
import numpy as np

def load_data(path="data/raw_sales_data.csv"):
    return pd.read_csv(path)

def clean(df):
    log = []
    log.append(f"Initial shape: {df.shape}")

    # 1. Drop fully empty rows
    before = len(df)
    df = df.dropna(how="all")
    log.append(f"Dropped {before - len(df)} fully empty rows")

    # 2. Drop rows with no OrderID (can't be tracked)
    before = len(df)
    df = df.dropna(subset=["OrderID"])
    log.append(f"Dropped {before - len(df)} rows with missing OrderID")

    # 3. Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    log.append(f"Dropped {before - len(df)} duplicate rows")

    # 4. Standardize text columns (strip whitespace, fix casing)
    df["Product"] = df["Product"].str.strip().str.title()
    df["Region"] = df["Region"].str.strip().str.title()
    df["Category"] = df["Category"].str.strip().str.title()
    df["PaymentMethod"] = df["PaymentMethod"].str.strip().str.title()
    df["PaymentMethod"] = df["PaymentMethod"].replace({"Cod": "COD"})

    # 5. Parse mixed-format dates
    def parse_date(val):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
            try:
                return pd.to_datetime(val, format=fmt)
            except (ValueError, TypeError):
                continue
        return pd.NaT
    df["Order Date"] = df["Order Date"].apply(parse_date)
    log.append(f"Unparseable dates: {df['Order Date'].isna().sum()}")

    # 6. Fix invalid Price (negative/zero/extreme outliers -> NaN then impute)
    df.loc[df["Price"] <= 0, "Price"] = np.nan
    q1, q3 = df["Price"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr
    outliers_price = (df["Price"] > upper_bound).sum()
    df.loc[df["Price"] > upper_bound, "Price"] = np.nan
    log.append(f"Flagged {outliers_price} price outliers -> set to NaN")
    df["Price"] = df.groupby("Product")["Price"].transform(lambda x: x.fillna(x.median()))

    # 7. Fix invalid Quantity (negative/zero/unrealistic bulk orders)
    df.loc[df["Quantity"] <= 0, "Quantity"] = np.nan
    df.loc[df["Quantity"] > 50, "Quantity"] = np.nan
    df["Quantity"] = df["Quantity"].fillna(df["Quantity"].median())
    df["Quantity"] = df["Quantity"].astype(int)

    # 8. Impute missing CustomerAge with median, cast to int
    df["CustomerAge"] = df["CustomerAge"].fillna(df["CustomerAge"].median()).astype(int)

    # 9. Fill missing PaymentMethod as 'Unknown'
    df["PaymentMethod"] = df["PaymentMethod"].fillna("Unknown")

    # 10. Fill missing Rating with category-level median
    df["Rating"] = df.groupby("Category")["Rating"].transform(lambda x: x.fillna(x.median()))

    # 11. Drop rows where Order Date still missing (can't analyze trends without it)
    before = len(df)
    df = df.dropna(subset=["Order Date"])
    log.append(f"Dropped {before - len(df)} rows with unparseable dates")

    # 12. Add derived column: Revenue
    df["Revenue"] = round(df["Price"] * df["Quantity"], 2)

    # 13. Final duplicate check on OrderID (keep first)
    before = len(df)
    df = df.drop_duplicates(subset=["OrderID"], keep="first")
    log.append(f"Dropped {before - len(df)} duplicate OrderIDs")

    df = df.reset_index(drop=True)
    log.append(f"Final shape: {df.shape}")
    return df, log

if __name__ == "__main__":
    df = load_data()
    cleaned, log = clean(df)
    cleaned.to_csv("data/cleaned_sales_data.csv", index=False)
    with open("data/cleaning_log.txt", "w") as f:
        f.write("\n".join(log))
    print("\n".join(log))
    print("\nMissing values after cleaning:")
    print(cleaned.isna().sum())
