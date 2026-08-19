"""
Generates a synthetic student performance dataset for the EDA task.
Simulates realistic relationships between study habits, background factors, and exam scores.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1200

gender = np.random.choice(["Male", "Female"], n)
parental_education = np.random.choice(
    ["High School", "Associate's Degree", "Bachelor's Degree", "Master's Degree"],
    n, p=[0.35, 0.30, 0.25, 0.10]
)
lunch = np.random.choice(["Standard", "Free/Reduced"], n, p=[0.65, 0.35])
test_prep = np.random.choice(["Completed", "Not Completed"], n, p=[0.4, 0.6])
study_hours_weekly = np.round(np.random.gamma(shape=3, scale=3, size=n), 1).clip(0, 40)
attendance_rate = np.round(np.random.beta(8, 2, n) * 100, 1)
extracurricular = np.random.choice(["Yes", "No"], n, p=[0.45, 0.55])
sleep_hours = np.round(np.random.normal(7, 1.2, n).clip(4, 10), 1)

edu_bonus = parental_education.copy()
edu_map = {"High School": 0, "Associate's Degree": 3, "Bachelor's Degree": 6, "Master's Degree": 9}
edu_score = np.vectorize(edu_map.get)(parental_education)

base_score = (
    50
    + study_hours_weekly * 1.3
    + attendance_rate * 0.15
    + edu_score
    + (test_prep == "Completed") * 6
    + (lunch == "Standard") * 4
    + np.where(sleep_hours < 6, -4, 0)
    + np.random.normal(0, 8, n)
)

math_score = np.clip(base_score + np.random.normal(0, 5, n), 0, 100).round(1)
reading_score = np.clip(base_score + np.random.normal(2, 5, n), 0, 100).round(1)
writing_score = np.clip(base_score + np.random.normal(-1, 5, n), 0, 100).round(1)

df = pd.DataFrame({
    "StudentID": [f"STU{3000+i}" for i in range(n)],
    "Gender": gender,
    "ParentalEducation": parental_education,
    "LunchType": lunch,
    "TestPrepCourse": test_prep,
    "StudyHoursWeekly": study_hours_weekly,
    "AttendanceRate": attendance_rate,
    "ExtracurricularActivity": extracurricular,
    "SleepHours": sleep_hours,
    "MathScore": math_score,
    "ReadingScore": reading_score,
    "WritingScore": writing_score,
})

df["AverageScore"] = df[["MathScore", "ReadingScore", "WritingScore"]].mean(axis=1).round(1)

df.to_csv("data/student_performance.csv", index=False)
print("Dataset created:", df.shape)
print(df.describe())
