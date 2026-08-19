"""
Preprocessing + training pipeline for the customer churn prediction task.
Trains Logistic Regression, Decision Tree, and Random Forest, then compares them.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              roc_curve, classification_report)
import joblib

def load_and_clean(path="data/raw_customer_data.csv"):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["CustomerID"], keep="first")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df

def preprocess(df):
    df = df.copy()
    encoders = {}
    cat_cols = ["ContractType", "InternetService", "TechSupport", "OnlineSecurity",
                "PaperlessBilling", "PaymentMethod"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    feature_cols = ["SeniorCitizen", "TenureMonths", "ContractType", "InternetService",
                     "TechSupport", "OnlineSecurity", "PaperlessBilling", "PaymentMethod",
                     "NumSupportCalls", "MonthlyCharges", "TotalCharges"]
    X = df[feature_cols]
    y = df["Churn"]
    return X, y, feature_cols, encoders

def train_and_evaluate():
    df = load_and_clean()
    X, y, feature_cols, encoders = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")
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
        predictions[name] = {"y_test": y_test, "y_pred": y_pred, "y_proba": y_proba}
        joblib.dump(model, f"models_{name.replace(' ', '_')}.pkl") if False else None

    results_df = pd.DataFrame(results).T.round(4)
    return results_df, predictions, models, X_test, y_test, feature_cols, scaler

if __name__ == "__main__":
    results_df, predictions, models, X_test, y_test, feature_cols, scaler = train_and_evaluate()
    print("Model Comparison:\n")
    print(results_df)
    results_df.to_csv("data/model_comparison_results.csv")

    best_model_name = results_df["ROC AUC"].idxmax()
    print(f"\nBest model by ROC AUC: {best_model_name}")

    print("\nClassification Report (Random Forest):")
    print(classification_report(predictions["Random Forest"]["y_test"], predictions["Random Forest"]["y_pred"]))
