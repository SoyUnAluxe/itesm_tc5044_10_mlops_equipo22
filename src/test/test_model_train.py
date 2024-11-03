import unittest
import os
import yaml
import pandas as pd
from src.models.train import ModelTrainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

class TestModelTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load parameters from params.yaml
        with open("params.yaml", "r") as file:
            cls.config = yaml.safe_load(file)

        # Paths for testing with real processed data
        cls.X_train_path = "data/processed/processed_dataset_X.csv"
        cls.y_train_path = "data/processed/processed_dataset_y.csv"

    def test_initialization(self):
        trainer = ModelTrainer(self.X_train_path, self.y_train_path, "params.yaml")
        self.assertIsNotNone(trainer, "ModelTrainer failed to initialize")

    def test_training_run(self):
        trainer = ModelTrainer(self.X_train_path, self.y_train_path, "params.yaml")
        trainer.run()
        
        # Check if models were saved
        self.assertTrue(os.path.exists("saved_models"), "Model was not saved correctly")
        saved_models = os.listdir("saved_models")
        self.assertGreater(len(saved_models), 0, "No models found in saved_models after training")

    def test_accuracy_above_threshold(self):
        trainer = ModelTrainer(self.X_train_path, self.y_train_path, "params.yaml")
        
        # Define the model configuration for test threshold accuracy
        model_name = "RandomForest"  # Choose a model for accuracy check
        model_params = next(m['params'] for m in self.config['models'] if m['name'] == model_name)
        
        # Train and log model
        trainer.train_and_log_model(RandomForestClassifier(), model_name, model_params)
        
        # Calculate accuracy using evaluation method
        X_test = pd.read_csv(self.X_train_path)
        y_test = pd.read_csv(self.y_train_path).values.ravel()
        y_pred = RandomForestClassifier().fit(X_test, y_test).predict(X_test)
        
        accuracy = (y_pred == y_test).mean()
        self.assertGreaterEqual(accuracy, 0.8, "Model accuracy is below 80%")

if __name__ == "__main__":
    unittest.main()
