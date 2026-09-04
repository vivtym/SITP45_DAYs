import os
import pickle
import pandas as pd

from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

app = Flask(__name__)

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"


def train_model():
    print("Training model for first run...")

    df = pd.read_csv(r"C:\Users\VIVEK MUDGAL\OneDrive\Documents\Desktop\CODING\Python\python\Grras\Deep_learning\Diabities_prediction\diabetes.csv")

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True)
    }

    best_model = None
    best_accuracy = 0
    best_name = ""

    for name, model in models.items():

        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, predictions)

        print(f"{name}: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name

    print(f"\nBest Model: {best_name}")
    print(f"Accuracy: {best_accuracy:.4f}")

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(best_model, f)

    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)

    print("Model Saved Successfully")


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


@app.route("/predict")
def predict():
    return render_template("predict.html")



@app.route("/result", methods=["POST"])
def result():

    try:

        pregnancies = float(request.form["pregnancies"])
        glucose = float(request.form["glucose"])
        blood_pressure = float(request.form["blood_pressure"])
        skin_thickness = float(request.form["skin_thickness"])
        insulin = float(request.form["insulin"])
        bmi = float(request.form["bmi"])
        dpf = float(request.form["dpf"])
        age = float(request.form["age"])

        data = [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]]

        scaled_data = scaler.transform(data)

        prediction = model.predict(scaled_data)[0]

        probability = model.predict_proba(scaled_data)[0][1]

        if prediction == 1:
            status = "Diabetic"
            advice = [
                "Consult a doctor",
                "Reduce sugar intake",
                "Exercise regularly",
                "Monitor blood glucose"
            ]
        else:
            status = "Non-Diabetic"
            advice = [
                "Maintain healthy diet",
                "Exercise regularly",
                "Stay hydrated",
                "Annual health checkup"
            ]

        return render_template(
            "result.html",
            status=status,
            probability=round(probability * 100, 2),
            advice=advice
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)