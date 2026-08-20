"""
Trains a loan default prediction model (Logistic Regression + Random Forest),
evaluates performance, and generates evaluation visuals.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, roc_curve, auc, classification_report)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

df = pd.read_csv("data/cleaned_loan_data.csv")

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

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
}

results = {}
predictions = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_proba)
    }
    predictions[name] = {"y_pred": y_pred, "y_proba": y_proba}

results_df = pd.DataFrame(results).T.round(4)
results_df.to_csv("data/model_results.csv")
print(results_df)

best = results_df["ROC AUC"].idxmax()
print(f"\nBest model by ROC AUC: {best}")
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, predictions["Random Forest"]["y_pred"]))

# --- Visuals ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, preds) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, preds["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Repaid", "Default"], yticklabels=["Repaid", "Default"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("images/confusion_matrices.png")
plt.close()

plt.figure(figsize=(7, 6))
for name, preds in predictions.items():
    fpr, tpr, _ = roc_curve(y_test, preds["y_proba"])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.title("ROC Curves — Loan Default Prediction")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("images/roc_curves.png")
plt.close()

rf = models["Random Forest"]
imp = sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: x[1], reverse=True)
names, vals = zip(*imp)
plt.figure(figsize=(8, 6))
sns.barplot(x=list(vals), y=list(names), hue=list(names), palette="viridis", legend=False)
plt.title("Feature Importance — Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("images/feature_importance.png")
plt.close()

print("\nModel evaluation visuals saved to images/")
