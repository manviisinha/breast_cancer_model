import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the pre-trained model
try:
    with open('breast_cancer.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print("⚠️ Model file 'breast_cancer.pkl' not found.")
except Exception as e:
    model = None
    print(f"⚠️ Error loading model: {e}")

# Route to display the form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction="⚠️ Model not loaded!")

    try:
        # Get features from the form
        mean_radius = float(request.form['mean_radius'])
        concave_points_mean = float(request.form['concave_points_mean'])
        perimeter_worst = float(request.form['perimeter_worst'])
        texture_worst = float(request.form['texture_worst'])

        # Make prediction
        features = np.array([mean_radius, concave_points_mean, perimeter_worst, texture_worst]).reshape(1, -1)
        prediction = model.predict(features)
        result = "Malignant" if prediction[0] == 1 else "Benign"

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"⚠️ Error: {str(e)}")

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

