from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the Trained Model
with open("XGB_Model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the Scaler
with open("Scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Function to Normalize Temperature values between -20 and 40 to the range 0-1
def normalize_temperature(temp):
    return (temp + 20) / 60  # Convert Temperature from range -20 to 40 to range 0-1

# Function to Convert Predicted Value back to its Original Scale
def inverse_transform_prediction(prediction):
    return np.expm1(prediction)  # Inverse Transformation of log1p

@app.route("/")
def home():
    return render_template("Index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get user Input from Form
    season = int(request.form["season"])
    hr = int(request.form["hr"])
    workingday = int(request.form["workingday"])
    weathersit = int(request.form["weathersit"])
    temp = float(request.form["temp"])
    
    # Normalize Temperature
    normalized_temp = normalize_temperature(temp)

    # Preprocess User Input
    input_data = np.array([[season, hr, workingday, weathersit, normalized_temp]])
    input_data_scaled = scaler.transform(input_data)

    # Make Prediction
    prediction = model.predict(input_data_scaled)
    
    # Convert Prediction back to Original Scale and Remove Decimals
    original_prediction = int(round(inverse_transform_prediction(prediction)[0]))

    # Return Prediction
    return render_template("Result.html", prediction=original_prediction)

if __name__ == "__main__":
    app.run(debug=True)
