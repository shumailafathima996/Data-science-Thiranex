"""
Generates evaluation visualizations: confusion matrices, ROC curves,
model comparison chart, and feature importance.
"""
import sys
sys.path.append("scripts")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from train_models import train_and_evaluate

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

results_df, predictions, models, X_test, y_test, feature_cols, scaler = train_and_evaluate()

# 1. Confusion matrices (one grid, 3 models)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, preds) in zip(axes, predictions.items()):
    cm = confusion_matrix(preds["y_test"], preds["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
fig.suptitle("Confusion Matrices by Model", fontsize=15, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("images/confusion_matrices.png")
plt.close()

# 2. ROC curves (all models overlaid)
plt.figure(figsize=(7, 6))
for name, preds in predictions.items():
    fpr, tpr, _ = roc_curve(preds["y_test"], preds["y_proba"])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
plt.title("ROC Curves — Model Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("images/roc_curves.png")
plt.close()

# 3. Model comparison bar chart
plt.figure(figsize=(9, 5))
results_plot = results_df.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
results_plot.columns = ["Model", "Metric", "Score"]
sns.barplot(data=results_plot, x="Metric", y="Score", hue="Model", palette="Set2")
plt.title("Model Performance Comparison")
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("images/model_comparison.png")
plt.close()

# 4. Feature importance (Random Forest)
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
imp_df = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
feat_names, feat_vals = zip(*imp_df)

plt.figure(figsize=(8, 6))
sns.barplot(x=list(feat_vals), y=list(feat_names), hue=list(feat_names), palette="viridis", legend=False)
plt.title("Feature Importance — Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("images/feature_importance.png")
plt.close()

# --- Combined dashboard ---
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3)

for i, (name, preds) in enumerate(predictions.items()):
    ax = fig.add_subplot(gs[0, i])
    cm = confusion_matrix(preds["y_test"], preds["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    ax.set_title(f"{name} — Confusion Matrix", fontsize=11)

ax_roc = fig.add_subplot(gs[1, 0])
for name, preds in predictions.items():
    fpr, tpr, _ = roc_curve(preds["y_test"], preds["y_proba"])
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, label=f"{name} ({roc_auc:.2f})", linewidth=2)
ax_roc.plot([0, 1], [0, 1], "--", color="gray")
ax_roc.set_title("ROC Curves")
ax_roc.legend(fontsize=8)

ax_bar = fig.add_subplot(gs[1, 1])
sns.barplot(data=results_plot, x="Metric", y="Score", hue="Model", palette="Set2", ax=ax_bar, legend=False)
ax_bar.set_title("Metric Comparison")
ax_bar.tick_params(axis='x', rotation=30)

ax_feat = fig.add_subplot(gs[1, 2])
sns.barplot(x=list(feat_vals), y=list(feat_names), hue=list(feat_names), palette="viridis", legend=False, ax=ax_feat)
ax_feat.set_title("Feature Importance (RF)")

fig.suptitle("Churn Prediction — Model Evaluation Dashboard", fontsize=17, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("images/dashboard.png", dpi=130)
plt.close()

print("All visualizations saved to images/")
print("\nFinal results:")
print(results_df)
