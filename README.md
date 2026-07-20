# SafeDrive-Insurance
These files form a web-based Safe Drive Internship Project. Source 2 generates and cleans a driver dataset to train TensorFlow models that predict driver risk and insurance premiums. Source 3 handles new prediction logic, Source 1 serves a Flask web dashboard visualizing the data, and Source 4 lists the python dependencies.

# Safe Drive - Driver Risk Assessment & Premium Prediction

A full-stack, machine learning-powered Flask web application designed to analyze driver behavior, predict insurance risk categories (Low, Medium, High), and estimate annual insurance premiums. 

This repository serves as a comprehensive internship project demonstrating data simulation, preprocessing, dual-architecture deep learning models, and interactive dashboard deployment.

## 📂 Project Structure

*   **`app.py`**: The core Flask web server. Handles user authentication (`admin`/`Admin@123`), manages session states, dynamically generates data visualizations (scatter plots, box plots, pie charts) using Matplotlib/Seaborn, and serves the interactive dashboard and prediction interface.
*   **`model_training.py`**: The data engineering and machine learning pipeline. Synthesizes a realistic 4,000-record dataset containing missing values, implements data cleaning (median/mode imputation), builds/compiles TensorFlow neural networks for dual targets (classification for risk, regression for premium), and saves the pipeline components.
*   **`prediction.py`**: A specialized module that loads the trained models and preprocessing pipelines to execute inference on real-time user inputs or batch queries, outputting calculated risk levels, percentage confidences, and premium amounts.
*   **`requirements.txt`**: Declares pin-point version-controlled dependencies required to run the pipeline, ensuring consistent reproduction across different production environments.

## 🚀 Tech Stack

*   **Backend Framework:** Flask (v3.0.3)
*   **Machine Learning / Deep Learning:** TensorFlow (v2.16.1), Scikit-Learn (v1.4.2)
*   **Data Processing:** Pandas (v2.2.2), NumPy (v1.26.4)
*   **Data Visualization:** Matplotlib (v3.8.4), Seaborn (v0.13.2)

## 📊 Machine Learning Architecture

The system trains two independent Neural Network architectures using `model_training.py`:
1.  **Risk Level Classifier**: A dense neural network outputting Softmax probabilities over 3 classes (`Low`, `Medium`, `High`) utilizing `sparse_categorical_crossentropy` loss.
2.  **Insurance Premium Regressor**: A dense neural network estimating the exact continuous premium amount optimized using `mean_absolute_error` (MAE).

Both architectures utilize `ReLU` hidden layers and input standardization via Scikit-Learn's `StandardScaler`.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/safe-drive-assessment.git
   cd safe-drive-assessment
   ```

2. **Set up a virtual environment and install dependencies:**
   ```bash
   python -bin/python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   python app.py
   ```
   *Note: On startup, `app.py` automatically validates the environment, generates the simulated dataset (`safe_drive_dataset.csv`), executes model training (`model.pkl`), builds static charts, and boots up the local Flask server at `http://127.0.0.1:5000`.*

## 🔒 Authentication
*   **Username:** `admin`
*   **Password:** `Admin@123`
