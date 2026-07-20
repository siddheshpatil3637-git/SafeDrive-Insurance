import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "safe_drive_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
FEATURE_COLUMNS = [
    "Age",
    "Experience_Years",
    "Average_Speed",
    "Harsh_Braking_Count",
    "Night_Driving_Hours",
    "Distance_Per_Day",
    "Mobile_Usage_Time",
    "Traffic_Violations",
    "Accident_History",
    "Driving_Score",
]


def generate_dataset(records=4000, output_path=DATASET_PATH):
    """Generate a realistic CSV dataset for the internship project."""
    np.random.seed(42)
    rows = []

    for driver_number in range(1, records + 1):
        age = int(np.random.randint(18, 71))
        max_experience = max(age - 18, 0)
        experience = int(np.random.randint(0, max_experience + 1)) if max_experience > 0 else 0
        average_speed = round(float(np.random.normal(58, 16)), 1)
        average_speed = min(max(average_speed, 25), 125)
        harsh_braking = int(np.random.poisson(4))
        night_hours = round(float(np.random.gamma(2, 1.6)), 1)
        night_hours = min(night_hours, 10)
        distance = round(float(np.random.normal(48, 24)), 1)
        distance = min(max(distance, 5), 170)
        mobile_usage = round(float(np.random.gamma(2, 5)), 1)
        mobile_usage = min(mobile_usage, 60)
        violations = int(np.random.poisson(0.8))
        accident_history = int(np.random.choice([0, 1, 2, 3, 4], p=[0.58, 0.24, 0.11, 0.05, 0.02]))

        score = 100
        score -= max(0, average_speed - 60) * 0.45
        score -= harsh_braking * 2.4
        score -= night_hours * 2.2
        score -= mobile_usage * 0.55
        score -= violations * 7
        score -= accident_history * 10
        score += min(experience, 20) * 0.7
        score += np.random.normal(0, 4)
        driving_score = int(min(max(round(score), 20), 100))

        if driving_score >= 75 and accident_history <= 1 and violations <= 1:
            risk_level = "Low"
        elif driving_score >= 50 and accident_history <= 2:
            risk_level = "Medium"
        else:
            risk_level = "High"

        risk_extra = {"Low": 0, "Medium": 3500, "High": 7500}[risk_level]
        premium = 6500 + risk_extra
        premium += max(0, 26 - age) * 95
        premium += max(0, average_speed - 70) * 45
        premium += harsh_braking * 120
        premium += night_hours * 90
        premium += mobile_usage * 35
        premium += violations * 750
        premium += accident_history * 1300
        premium -= min(experience, 15) * 120
        premium += np.random.normal(0, 450)
        premium = round(float(max(premium, 3500)), 2)

        rows.append(
            [
                f"DRV{driver_number:05d}",
                age,
                experience,
                average_speed,
                harsh_braking,
                night_hours,
                distance,
                mobile_usage,
                violations,
                accident_history,
                driving_score,
                risk_level,
                premium,
            ]
        )

    columns = [
        "Driver_ID",
        "Age",
        "Experience_Years",
        "Average_Speed",
        "Harsh_Braking_Count",
        "Night_Driving_Hours",
        "Distance_Per_Day",
        "Mobile_Usage_Time",
        "Traffic_Violations",
        "Accident_History",
        "Driving_Score",
        "Risk_Level",
        "Insurance_Premium",
    ]
    dataset = pd.DataFrame(rows, columns=columns)

    # Add a few missing values so the cleaning step has real work to do.
    for column in ["Average_Speed", "Mobile_Usage_Time", "Driving_Score"]:
        missing_indexes = dataset.sample(frac=0.01, random_state=42).index
        dataset.loc[missing_indexes, column] = np.nan

    dataset.to_csv(output_path, index=False)
    return dataset


def load_and_clean_dataset(dataset_path=DATASET_PATH):
    """Load CSV data, clean missing values, and return a ready DataFrame."""
    if not os.path.exists(dataset_path):
        generate_dataset(output_path=dataset_path)

    data = pd.read_csv(dataset_path)
    data = data.drop_duplicates()

    numeric_columns = data.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        data[column] = data[column].fillna(data[column].median())

    data["Risk_Level"] = data["Risk_Level"].fillna(data["Risk_Level"].mode()[0])
    return data


def build_risk_model(input_count):
    """Create a small TensorFlow classification model."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_count,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_premium_model(input_count):
    """Create a small TensorFlow regression model."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_count,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_absolute_error", metrics=["mae"])
    return model


def train_and_save_model():
    """Train both models and save them with Pickle."""
    data = load_and_clean_dataset()

    x = data[FEATURE_COLUMNS]
    y_risk = data["Risk_Level"]
    y_premium = data["Insurance_Premium"]

    label_encoder = LabelEncoder()
    y_risk_encoded = label_encoder.fit_transform(y_risk)

    x_train, x_test, y_risk_train, y_risk_test, y_premium_train, y_premium_test = train_test_split(
        x,
        y_risk_encoded,
        y_premium,
        test_size=0.2,
        random_state=42,
        stratify=y_risk_encoded,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    risk_model = build_risk_model(x_train_scaled.shape[1])
    risk_model.fit(x_train_scaled, y_risk_train, epochs=25, batch_size=32, verbose=0)
    risk_loss, risk_accuracy = risk_model.evaluate(x_test_scaled, y_risk_test, verbose=0)

    premium_model = build_premium_model(x_train_scaled.shape[1])
    premium_model.fit(x_train_scaled, y_premium_train, epochs=35, batch_size=32, verbose=0)
    premium_loss, premium_mae = premium_model.evaluate(x_test_scaled, y_premium_test, verbose=0)

    saved_items = {
        "scaler": scaler,
        "label_encoder": label_encoder,
        "risk_model": risk_model,
        "premium_model": premium_model,
        "feature_columns": FEATURE_COLUMNS,
        "risk_accuracy": float(risk_accuracy),
        "premium_mae": float(premium_mae),
    }

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(saved_items, file)

    print("Dataset and model files are ready.")
    print(f"Risk model accuracy: {risk_accuracy:.2f}")
    print(f"Premium model MAE: {premium_mae:.2f}")


if __name__ == "__main__":
    train_and_save_model()
