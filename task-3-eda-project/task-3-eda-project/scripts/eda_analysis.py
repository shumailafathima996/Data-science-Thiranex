"""
Exploratory Data Analysis script: statistical summaries, correlations,
and key visualizations for the student performance dataset.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

df = pd.read_csv("data/student_performance.csv")

# --- Statistical summary ---
summary = df.describe(include="all").T
summary.to_csv("data/statistical_summary.csv")

numeric_cols = ["StudyHoursWeekly", "AttendanceRate", "SleepHours",
                 "MathScore", "ReadingScore", "WritingScore", "AverageScore"]
corr = df[numeric_cols].corr()
corr.to_csv("data/correlation_matrix.csv")

print("Correlation with AverageScore:")
print(corr["AverageScore"].sort_values(ascending=False))

# 1. Correlation heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap — Numeric Features")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png")
plt.close()

# 2. Score distributions
plt.figure(figsize=(9, 5))
for col, color in zip(["MathScore", "ReadingScore", "WritingScore"], ["#2E86AB", "#F18F01", "#A23B72"]):
    sns.kdeplot(df[col], label=col, fill=True, alpha=0.2, color=color)
plt.title("Score Distributions by Subject")
plt.xlabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig("images/score_distributions.png")
plt.close()

# 3. Study hours vs Average score (scatter + trend)
plt.figure(figsize=(8, 5))
sns.regplot(data=df, x="StudyHoursWeekly", y="AverageScore",
            scatter_kws={"alpha": 0.3, "s": 15}, line_kws={"color": "red"})
plt.title("Study Hours vs Average Score")
plt.tight_layout()
plt.savefig("images/study_hours_vs_score.png")
plt.close()

# 4. Average score by parental education
plt.figure(figsize=(8, 5))
edu_order = ["High School", "Associate's Degree", "Bachelor's Degree", "Master's Degree"]
sns.boxplot(data=df, x="ParentalEducation", y="AverageScore", order=edu_order,
            hue="ParentalEducation", palette="viridis", legend=False)
plt.title("Average Score by Parental Education Level")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("images/score_by_parental_education.png")
plt.close()

# 5. Test prep course impact
plt.figure(figsize=(7, 5))
sns.violinplot(data=df, x="TestPrepCourse", y="AverageScore", order=["Completed", "Not Completed"],
               hue="TestPrepCourse", hue_order=["Completed", "Not Completed"], palette="Set2", legend=False)
plt.title("Average Score: Test Prep Course Completed vs None")
plt.tight_layout()
plt.savefig("images/test_prep_impact.png")
plt.close()

# 6. Attendance vs score, colored by lunch type
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="AttendanceRate", y="AverageScore", hue="LunchType",
                 alpha=0.5, palette="Set1", s=25)
plt.title("Attendance Rate vs Average Score by Lunch Type")
plt.tight_layout()
plt.savefig("images/attendance_vs_score.png")
plt.close()

# --- Combined dashboard ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[0,0], cbar=False)
axes[0,0].set_title("Correlation Heatmap")

for col, color in zip(["MathScore", "ReadingScore", "WritingScore"], ["#2E86AB", "#F18F01", "#A23B72"]):
    sns.kdeplot(df[col], label=col, fill=True, alpha=0.2, color=color, ax=axes[0,1])
axes[0,1].set_title("Score Distributions")
axes[0,1].legend(fontsize=8)

sns.regplot(data=df, x="StudyHoursWeekly", y="AverageScore",
            scatter_kws={"alpha": 0.3, "s": 10}, line_kws={"color": "red"}, ax=axes[0,2])
axes[0,2].set_title("Study Hours vs Avg Score")

sns.boxplot(data=df, x="ParentalEducation", y="AverageScore", order=edu_order,
            hue="ParentalEducation", palette="viridis", legend=False, ax=axes[1,0])
axes[1,0].set_title("Score by Parental Education")
axes[1,0].tick_params(axis='x', rotation=20)

sns.violinplot(data=df, x="TestPrepCourse", y="AverageScore", order=["Completed", "Not Completed"],
               hue="TestPrepCourse", hue_order=["Completed", "Not Completed"], palette="Set2", legend=False, ax=axes[1,1])
axes[1,1].set_title("Test Prep Course Impact")

sns.scatterplot(data=df, x="AttendanceRate", y="AverageScore", hue="LunchType",
                 alpha=0.5, palette="Set1", s=20, ax=axes[1,2])
axes[1,2].set_title("Attendance vs Score by Lunch Type")

fig.suptitle("Student Performance — EDA Dashboard", fontsize=17, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("images/dashboard.png", dpi=130)
plt.close()

print("\nAll visualizations saved to images/")
