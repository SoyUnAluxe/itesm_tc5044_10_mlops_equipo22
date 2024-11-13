from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import sys, os


from src.data.preprocess import DataPreprocessor
data_preprocessor = DataPreprocessor(raw_data_path='', processed_data_path='')

app = Flask(__name__)

# Cargar el modelo preentrenado
model = joblib.load('saved_models/ml_models/RandomForest_20241113_125357.pkl')
le = joblib.load("saved_models/label_encoder.pkl")

# Preprocesamiento, similar a DataPreprocessor
def preprocess_data(data):
   return data_preprocessor.preprocess(train=False, processed_data=data)

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener datos del request
    input_data = request.get_json()
    data_df = pd.DataFrame([input_data])
    
    # Preprocesar los datos
    X,_ = preprocess_data(data_df)

    prediction = model.predict(X)
    label = le.inverse_transform(prediction)[0]
    
    # Codificar la respuesta
    response = {'prediction': label}
    return jsonify(response)

@app.route('/test', methods=['POST'])
def test():
    return "OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
