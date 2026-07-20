import pickle

import numpy as np


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


def load_saved_model(model_path="model.pkl"):
    """Load the saved TensorFlow models and preprocessing objects."""
    with open(model_path, "rb") as file:
        return pickle.load(file)


def predict_driver_risk(input_data, model_path="model.pkl"):
    """Predict risk level and premium for one driver's details."""
    saved_items = load_saved_model(model_path)
    scaler = saved_items["scaler"]
    label_encoder = saved_items["label_encoder"]
    risk_model = saved_items["risk_model"]
    premium_model = saved_items["premium_model"]

    values = [float(input_data[column]) for column in FEATURE_COLUMNS]
    input_array = np.array([values])
    scaled_input = scaler.transform(input_array)

    risk_probabilities = risk_model.predict(scaled_input, verbose=0)[0]
    risk_index = int(np.argmax(risk_probabilities))
    risk_label = label_encoder.inverse_transform([risk_index])[0]
    premium_prediction = premium_model.predict(scaled_input, verbose=0)[0][0]

    return {
        "risk_level": risk_label,
        "risk_confidence": round(float(risk_probabilities[risk_index]) * 100, 2),
        "estimated_premium": round(float(premium_prediction), 2),
    }
