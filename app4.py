from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__, template_folder='template')

# Load the trained model
model_path = "heartrate_classification_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please make sure the model file exists.")

@app.route('/')
def index():
    return render_template('C:/Users/atber/Downloads/heart rate proj/templates\index.html')

@app.route('/predict', methods=['POST'])    
def predict():
    try:
        # Get input data from the request
        data = request.get_json(force=True)
        
        # Convert JSON data to a pandas DataFrame
        input_data = pd.DataFrame(data)
        
        # Perform prediction
        prediction = int(model.predict(input_data)[0])
                      
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)