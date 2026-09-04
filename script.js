import os
import pickle

import numpy as np
import pandas as pd

from flask import Flask, render_template, request

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"
DATA_FILE = "diabetes.csv"


def train_model():
    print("Training model for first time...")

    df = pd.read_csv(DATA_FILE)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier()
    }

    best_model = None
    best_accuracy = 0

    print("\nModel Accuracy\n")

    for name, model in models.items():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)

        print(f"{name}: {acc:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model

    print("\nBest Model Selected:", best_model.__class__.__name__)
    print("Accuracy:", round(best_accuracy * 100, 2), "%")

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(best_model, f)

    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)


if not os.path.exists(MODEL_FILE):

    train_model()


with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

with open(SCALER_FILE, "rb") as f:
    scaler = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        data = [
            float(request.form["Pregnancies"]),
            float(request.form["Glucose"]),
            float(request.form["BloodPressure"]),
            float(request.form["SkinThickness"]),
            float(request.form["Insulin"]),
            float(request.form["BMI"]),
            float(request.form["DiabetesPedigreeFunction"]),
            float(request.form["Age"])
        ]

        data = np.array(data).reshape(1, -1)

        data = scaler.transform(data)

        prediction = model.predict(data)[0]

        probability = model.predict_proba(data)[0]

        confidence = round(max(probability) * 100, 2)

        return render_template(
            "result.html",
            prediction=prediction,
            probability=confidence
        )

    return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True)