from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import keras

app = Flask(__name__)

# Load CSV files
careers_df = pd.read_csv("Careers.csv")
engineer_personality_df = pd.read_csv("engineer_personality_dataset.csv")

# Load models
career_model = keras.models.load_model("career_model.keras")
engineer_model = keras.models.load_model("career_engineer_model.keras")

# Prepare label encoders for both datasets
career_label_encoder = LabelEncoder()
career_label_encoder.fit(careers_df.iloc[:, -1])  # Last column is the output for career

engineer_label_encoder = LabelEncoder()
engineer_label_encoder.fit(engineer_personality_df.iloc[:, -1])  # Last column is the output for engineers

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data for the personality-based test
    realistic = float(request.form['realistic'])
    investigative = float(request.form['investigative'])
    artistic = float(request.form['artistic'])
    social = float(request.form['social'])
    enterprising = float(request.form['enterprising'])
    conventional = float(request.form['conventional'])

    # Predict career using the career model
    user_input = np.array([[realistic, investigative, artistic, social, enterprising, conventional]])
    prediction = career_model.predict(user_input)
    predicted_class = np.argmax(prediction)
    career_prediction = career_label_encoder.inverse_transform([predicted_class])[0]

    # If the prediction is "Engineer", redirect to the engineer personality test
    if career_prediction == "Engineer":
        return redirect(url_for('engineer_test'))

    # Otherwise, show the career prediction
    return render_template('index.html', prediction=career_prediction)

@app.route('/engineer_test')
def engineer_test():
    return render_template('engineer_test.html')

@app.route('/engineer_predict', methods=['POST'])
def engineer_predict():
    # Collect form data for the engineer-specific test
    manipulation_capacity = float(request.form['manipulation'])
    analytical_capacity = float(request.form['analytical'])
    computer_skills = float(request.form['computer'])
    teamwork = float(request.form['teamwork'])
    self_learning = float(request.form['selflearning'])

    # Predict using the engineer personality model
    user_input = np.array([[manipulation_capacity, analytical_capacity, computer_skills, teamwork, self_learning]])
    prediction = engineer_model.predict(user_input)
    predicted_class = np.argmax(prediction)
    engineer_prediction = engineer_label_encoder.inverse_transform([predicted_class])[0]

    # Show the final engineer prediction
    return render_template('engineer_result.html', prediction=engineer_prediction)

if __name__ == "__main__":
    app.run(debug=True)
