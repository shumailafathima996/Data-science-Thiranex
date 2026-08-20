"""
Combined dashboard tying together EDA and model evaluation results.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 130

df = pd.read_csv("data/cleaned_loan_data.csv")

# Re-run quick model fit for dashboard (mirrors train_model.py)
cat_cols = ["EmploymentType", "HomeOwnership", "MaritalStatus", "LoanPurpose"]
df_model = df.copy()
for col in cat_cols:
    df_model[col] = LabelEncoder().fit_transform(df_model[col])
feature_cols = ["Age", "AnnualIncome", "EmploymentType", "CreditScore", "LoanAmount",
                 "LoanTermMonths", "NumExistingLoans", "HomeOwnership", "MaritalStatus",
                 "LoanPurpose", "DebtToIncomeRatio"]
X = df_model[feature_cols]
y = df_model["LoanDefault"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
log_reg.fit(X_train_scaled, y_train)
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)

fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 3)

# 1. Correlation heatmap
numeric_cols = ["Age", "AnnualIncome", "CreditScore", "LoanAmount",
                 "LoanTermMonths", "NumExistingLoans", "DebtToIncomeRatio", "LoanDefault"]
corr = df[numeric_cols].corr()
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar=False, ax=ax1,
            annot_kws={"size": 7})
ax1.set_title("Correlation Heatmap", fontsize=11)
ax1.tick_params(labelsize=7)

# 2. Default rate by credit band
df["CreditScoreBand"] = pd.cut(df["CreditScore"], bins=[300, 580, 670, 740, 800, 850],
                                labels=["Poor", "Fair", "Good", "V.Good", "Excellent"])
rate_by_band = df.groupby("CreditScoreBand", observed=True)["LoanDefault"].mean()
ax2 = fig.add_subplot(gs[0, 1])
sns.barplot(x=rate_by_band.index, y=rate_by_band.values, hue=rate_by_band.index, palette="RdYlGn", legend=False, ax=ax2)
ax2.set_title("Default Rate by Credit Band", fontsize=11)

# 3. DTI by outcome
ax3 = fig.add_subplot(gs[0, 2])
sns.boxplot(data=df, x="LoanDefault", y="DebtToIncomeRatio", hue="LoanDefault", palette="Set2", legend=False, ax=ax3)
ax3.set_xticks([0, 1])
ax3.set_xticklabels(["Repaid", "Default"])
ax3.set_title("Debt-to-Income by Outcome", fontsize=11)

# 4. Confusion matrix (Random Forest)
y_pred_rf = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_rf)
ax4 = fig.add_subplot(gs[1, 0])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax4,
            xticklabels=["Repaid", "Default"], yticklabels=["Repaid", "Default"])
ax4.set_title("Random Forest — Confusion Matrix", fontsize=11)

# 5. ROC curves
ax5 = fig.add_subplot(gs[1, 1])
for name, model, Xte in [("Logistic Regression", log_reg, X_test_scaled), ("Random Forest", rf, X_test)]:
    proba = model.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    ax5.plot(fpr, tpr, label=f"{name} ({roc_auc:.2f})", linewidth=2)
ax5.plot([0, 1], [0, 1], "--", color="gray")
ax5.set_title("ROC Curves", fontsize=11)
ax5.legend(fontsize=8)

# 6. Feature importance
imp = sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: x[1], reverse=True)
names, vals = zip(*imp)
ax6 = fig.add_subplot(gs[1, 2])
sns.barplot(x=list(vals), y=list(names), hue=list(names), palette="viridis", legend=False, ax=ax6)
ax6.set_title("Feature Importance (RF)", fontsize=11)
ax6.tick_params(labelsize=8)

fig.suptitle("Loan Default Risk — Real-World Finance Project Dashboard", fontsize=17, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("images/dashboard.png", dpi=130)
plt.close()
print("Dashboard saved to images/dashboard.png")
