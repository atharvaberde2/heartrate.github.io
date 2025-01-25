from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load your ML model (make sure to provide the correct path to your model)
with open('heartrate_classification_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Extract features from the request data
    features = [
        data['age'],
        data['sex'],
        data['chest pain type'],
        data['excercise angina'],
        data['resting bp s'],
        data['cholesterol'],
        data['resting ecg'],
        data['ST slope'],
        data['oldpeak'],
        data['max heart rate']
    ]
    
    # Convert features to numpy array and reshape for prediction
    features = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    
    # Return the prediction (you can customize this response)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
