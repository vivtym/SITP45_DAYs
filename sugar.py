import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

warnings.filterwarnings("ignore")

# ===========================
# Load Dataset
# ===========================

df = pd.read_csv(r"C:\Users\VIVEK MUDGAL\OneDrive\Documents\Desktop\CODING\Python\python\Grras\Deep_learning\Diabities_prediction\diabetes.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

# ===========================
# Features and Target
# ===========================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ===========================
# Train Test Split
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ===========================
# Feature Scaling
# ===========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===========================
# Logistic Regression
# ===========================

log_model = LogisticRegression(random_state=42)

log_model.fit(X_train_scaled, y_train)

log_pred = log_model.predict(X_test_scaled)

log_acc = accuracy_score(y_test, log_pred)
log_recall = recall_score(y_test, log_pred)
log_precision = precision_score(y_test, log_pred)

print("\n========== Logistic Regression ==========")
print(f"Accuracy : {log_acc*100:.2f}%")
print(f"Recall   : {log_recall:.2f}")
print(f"Precision: {log_precision:.2f}")

# ===========================
# Decision Tree
# ===========================

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)

print("\n========== Decision Tree ==========")
print(f"Accuracy : {dt_acc*100:.2f}%")
print(f"Recall   : {dt_recall:.2f}")
print(f"Precision: {dt_precision:.2f}")

# ===========================
# Random Forest
# ===========================

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)

print("\n========== Random Forest ==========")
print(f"Accuracy : {rf_acc*100:.2f}%")
print(f"Recall   : {rf_recall:.2f}")
print(f"Precision: {rf_precision:.2f}")

# ===========================
# Accuracy Comparison
# ===========================

models = [
    "Logistic\nRegression",
    "Decision\nTree",
    "Random\nForest"
]

accuracy = [
    log_acc,
    dt_acc,
    rf_acc
]

plt.figure(figsize=(8,5))
plt.bar(models, accuracy)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.show()

# ===========================
# Metric Comparison
# ===========================

metrics = ["Accuracy", "Recall", "Precision"]

log_values = [log_acc, log_recall, log_precision]
dt_values = [dt_acc, dt_recall, dt_precision]
rf_values = [rf_acc, rf_recall, rf_precision]

plt.figure(figsize=(10,5))

plt.plot(metrics, log_values, marker='o', label="Logistic Regression")
plt.plot(metrics, dt_values, marker='o', label="Decision Tree")
plt.plot(metrics, rf_values, marker='o', label="Random Forest")

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0,1)
plt.legend()
plt.grid(True)

plt.show()

# ===========================
# User Prediction
# ===========================

print("\n========== Diabetes Prediction ==========")

Pregnancies = int(input("Enter Pregnancies: "))
Glucose = int(input("Enter Glucose Level: "))
BloodPressure = int(input("Enter Blood Pressure: "))
SkinThickness = int(input("Enter Skin Thickness: "))
Insulin = int(input("Enter Insulin Level: "))
BMI = float(input("Enter BMI: "))
DiabetesPedigreeFunction = float(input("Enter Diabetes Pedigree Function: "))
Age = int(input("Enter Age: "))

user_data = [[
    Pregnancies,
    Glucose,
    BloodPressure,
    SkinThickness,
    Insulin,
    BMI,
    DiabetesPedigreeFunction,
    Age
]]

# Scale input for Logistic Regression
user_scaled = scaler.transform(user_data)

prediction = log_model.predict(user_scaled)

print("\nPrediction Result")

if prediction[0] == 1:
    print("🩺 The patient is likely to have Diabetes.")
else:
    print("✅ The patient is not likely to have Diabetes.")