import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
import yaml

class ModelEvaluator:
    def __init__(self, X_test_path, y_test_path, config_path):
        self.X_test_path = X_test_path
        self.y_test_path = y_test_path
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"{self.config_path} not found.")
        with open(self.config_path, 'r') as conf_file:
            config = yaml.safe_load(conf_file)
        return config

    def load_data(self):
        if not os.path.exists(self.X_test_path) or not os.path.exists(self.y_test_path):
            raise FileNotFoundError(f"Data files {self.X_test_path} or {self.y_test_path} not found.")
        X_test = pd.read_csv(self.X_test_path)
        y_test = pd.read_csv(self.y_test_path).values.ravel() 
        return X_test, y_test

    def load_model(self):
        if not os.path.exists(self.config['reports']['model']):
            raise FileNotFoundError(f"Model file {self.config['reports']['model']} not found.")
        self.model = load_model(self.config['reports']['model'])
        return self.model

    def evaluate(self):
        X_test, y_test = self.load_data()
        model = self.load_model()

        print("Evaluating the loaded model...")
        y_pred = np.argmax(model.predict(X_test), axis=self.config['train']['axis'])
        accuracy = accuracy_score(y_test, y_pred)

        print(f'Accuracy del modelo cargado: {accuracy:.4f}')
         # Guardar el resultado en el archivo de reporte
        with open("reports/evaluation.txt", 'w') as report_file:
            report_file.write(f'Accuracy del modelo cargado: {accuracy:.4f}\n')

        return accuracy

    def run(self):
        print("Starting the evaluation process...")
        self.evaluate()
        print("Evaluation completed.")

if __name__ == "__main__":
    evaluator = ModelEvaluator(r'data/processed/processed_dataset_X.csv', 
                               r'data/processed/processed_dataset_y.csv', 
                               'params.yaml')
    evaluator.run()
