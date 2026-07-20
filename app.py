import os
from functools import wraps

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, redirect, render_template, request, send_file, session, url_for

from model_training import DATASET_PATH, MODEL_PATH, FEATURE_COLUMNS, generate_dataset, train_and_save_model
from prediction import predict_driver_risk


app = Flask(__name__)
app.secret_key = "safe-drive-internship-secret-key"

USERNAME = "admin"
PASSWORD = "Admin@123"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_FOLDER = os.path.join(BASE_DIR, "static", "images")


def prepare_project_files():
    """Create dataset, model, and chart files when the project starts."""
    if not os.path.exists(DATASET_PATH):
        generate_dataset(output_path=DATASET_PATH)

    if not os.path.exists(MODEL_PATH):
        train_and_save_model()

    os.makedirs(CHART_FOLDER, exist_ok=True)
    create_charts()


def load_dataset():
    """Read the CSV dataset used by the dashboard."""
    return pd.read_csv(DATASET_PATH)


def create_charts():
    """Create Matplotlib and Seaborn charts for the dashboard."""
    data = load_dataset()
    sns.set_theme(style="whitegrid")

    charts = [
        ("age_premium.png", lambda: sns.scatterplot(data=data, x="Age", y="Insurance_Premium", hue="Risk_Level")),
        ("speed_risk.png", lambda: sns.boxplot(data=data, x="Risk_Level", y="Average_Speed")),
        ("violations.png", lambda: sns.countplot(data=data, x="Traffic_Violations", hue="Risk_Level")),
        ("accident_history.png", lambda: sns.countplot(data=data, x="Accident_History", hue="Risk_Level")),
        ("score_distribution.png", lambda: sns.histplot(data=data, x="Driving_Score", hue="Risk_Level", bins=25, kde=True)),
    ]

    for file_name, chart_function in charts:
        plt.figure(figsize=(8, 5))
        chart_function()
        plt.tight_layout()
        plt.savefig(os.path.join(CHART_FOLDER, file_name), dpi=120)
        plt.close()

    plt.figure(figsize=(6, 6))
    risk_counts = data["Risk_Level"].value_counts()
    plt.pie(risk_counts.values, labels=risk_counts.index, autopct="%1.1f%%", startangle=90)
    plt.title("Risk Category Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_FOLDER, "risk_pie.png"), dpi=120)
    plt.close()


def login_required(route_function):
    """Protect dashboard pages from users who are not logged in."""
    @wraps(route_function)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return route_function(*args, **kwargs)

    return wrapper


@app.route("/", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == USERNAME and password == PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("dashboard"))

        error = "Invalid username or password."

    return render_template("login.html", error=error)


@app.route("/dashboard")
@login_required
def dashboard():
    data = load_dataset()
    summary = {
        "total_drivers": len(data),
        "average_premium": round(data["Insurance_Premium"].mean(), 2),
        "high_risk": int((data["Risk_Level"] == "High").sum()),
        "low_risk": int((data["Risk_Level"] == "Low").sum()),
        "average_score": round(data["Driving_Score"].mean(), 1),
    }

    sample_driver = data.sample(1, random_state=10).iloc[0]
    sample_input = {column: sample_driver[column] for column in FEATURE_COLUMNS}
    sample_prediction = predict_driver_risk(sample_input, MODEL_PATH)

    return render_template("dashboard.html", summary=summary, sample_prediction=sample_prediction)


@app.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction():
    result = None
    form_data = {}

    if request.method == "POST":
        form_data = {
            "Age": request.form.get("Age", 0),
            "Experience_Years": request.form.get("Experience_Years", 0),
            "Average_Speed": request.form.get("Average_Speed", 0),
            "Harsh_Braking_Count": request.form.get("Harsh_Braking_Count", 0),
            "Night_Driving_Hours": request.form.get("Night_Driving_Hours", 0),
            "Distance_Per_Day": request.form.get("Distance_Per_Day", 0),
            "Mobile_Usage_Time": request.form.get("Mobile_Usage_Time", 0),
            "Traffic_Violations": request.form.get("Traffic_Violations", 0),
            "Accident_History": request.form.get("Accident_History", 0),
            "Driving_Score": request.form.get("Driving_Score", 0),
        }
        result = predict_driver_risk(form_data, MODEL_PATH)

    return render_template("prediction.html", result=result, form_data=form_data)


@app.route("/about")
@login_required
def about():
    return render_template("about.html")


@app.route("/download-dataset")
@login_required
def download_dataset():
    return send_file(DATASET_PATH, as_attachment=True)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    prepare_project_files()
    app.run(debug=True)
