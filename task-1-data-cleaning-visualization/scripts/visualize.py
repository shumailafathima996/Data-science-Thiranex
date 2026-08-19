"""
Visualization script: generates key charts + a combined dashboard image
from the cleaned sales dataset.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

df = pd.read_csv("data/cleaned_sales_data.csv", parse_dates=["Order Date"])
df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

# 1. Revenue by Category
plt.figure(figsize=(8, 5))
cat_rev = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
sns.barplot(x=cat_rev.values, y=cat_rev.index, hue=cat_rev.index, palette="viridis", legend=False)
plt.title("Total Revenue by Category")
plt.xlabel("Revenue ($)")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig("images/revenue_by_category.png")
plt.close()

# 2. Monthly Revenue Trend
plt.figure(figsize=(9, 5))
monthly = df.groupby("Month")["Revenue"].sum().sort_index()
sns.lineplot(x=monthly.index, y=monthly.values, marker="o", color="#2E86AB")
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/monthly_revenue_trend.png")
plt.close()

# 3. Orders by Region
plt.figure(figsize=(7, 5))
region_counts = df["Region"].value_counts()
sns.barplot(x=region_counts.index, y=region_counts.values, hue=region_counts.index, palette="crest", legend=False)
plt.title("Number of Orders by Region")
plt.xlabel("Region")
plt.ylabel("Order Count")
plt.tight_layout()
plt.savefig("images/orders_by_region.png")
plt.close()

# 4. Price distribution (before vs after cleaning conceptually - here just cleaned)
plt.figure(figsize=(8, 5))
sns.histplot(df["Price"], bins=30, kde=True, color="#F18F01")
plt.title("Distribution of Product Price (Cleaned Data)")
plt.xlabel("Price ($)")
plt.tight_layout()
plt.savefig("images/price_distribution.png")
plt.close()

# 5. Payment Method breakdown
plt.figure(figsize=(7, 7))
pay_counts = df["PaymentMethod"].value_counts()
plt.pie(pay_counts.values, labels=pay_counts.index, autopct="%1.1f%%",
        colors=sns.color_palette("Set2"), startangle=90)
plt.title("Payment Method Breakdown")
plt.tight_layout()
plt.savefig("images/payment_method_breakdown.png")
plt.close()

# 6. Rating vs Category boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Category", y="Rating", hue="Category", palette="pastel", legend=False)
plt.title("Customer Rating Distribution by Category")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("images/rating_by_category.png")
plt.close()

# --- Combined dashboard (2x3 grid) ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

sns.barplot(x=cat_rev.values, y=cat_rev.index, hue=cat_rev.index, palette="viridis", legend=False, ax=axes[0,0])
axes[0,0].set_title("Revenue by Category")

sns.lineplot(x=monthly.index, y=monthly.values, marker="o", color="#2E86AB", ax=axes[0,1])
axes[0,1].set_title("Monthly Revenue Trend")
axes[0,1].tick_params(axis='x', rotation=45)

sns.barplot(x=region_counts.index, y=region_counts.values, hue=region_counts.index, palette="crest", legend=False, ax=axes[0,2])
axes[0,2].set_title("Orders by Region")

sns.histplot(df["Price"], bins=30, kde=True, color="#F18F01", ax=axes[1,0])
axes[1,0].set_title("Price Distribution")

pay_counts_plot = df["PaymentMethod"].value_counts()
axes[1,1].pie(pay_counts_plot.values, labels=pay_counts_plot.index, autopct="%1.0f%%",
              colors=sns.color_palette("Set2"), startangle=90)
axes[1,1].set_title("Payment Method Breakdown")

sns.boxplot(data=df, x="Category", y="Rating", hue="Category", palette="pastel", legend=False, ax=axes[1,2])
axes[1,2].set_title("Rating by Category")
axes[1,2].tick_params(axis='x', rotation=20)

fig.suptitle("Sales Data Dashboard", fontsize=18, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("images/dashboard.png", dpi=130)
plt.close()

print("All visualizations saved to images/")
