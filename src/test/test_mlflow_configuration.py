import unittest
import os
import yaml
import mlflow

class TestMLflowConfiguration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the configuration from the params.yaml file
        cls.config_path = 'params.yaml'
        # Check if the configuration file exists
        if not os.path.exists(cls.config_path):
            raise FileNotFoundError(f"{cls.config_path} not found.")
        with open(cls.config_path, 'r') as file:
            cls.config = yaml.safe_load(file)  # Load the YAML file content

    def test_mlflow_configuration(self):
        # Verify that the 'mlflow' configuration section exists in params.yaml
        mlflow_config = self.config.get('mlflow', {})
        
        # Check if 'experiment_name' key is present in the 'mlflow' section
        self.assertIn('experiment_name', mlflow_config, "'experiment_name' must be present in the MLflow configuration")
        
        # Ensure that 'experiment_name' is a string
        self.assertIsInstance(mlflow_config['experiment_name'], str, "'experiment_name' must be a string")
        
        # Check if MLflow tracking URI is set correctly
        tracking_uri = mlflow.get_tracking_uri()
        # Assert that tracking URI is not None
        self.assertIsNotNone(tracking_uri, "MLflow tracking URI is not configured properly.")
        
        # Extract the base path from the URI
        base_path = os.path.basename(tracking_uri)

        # Assert that the base path is 'mlruns'
        self.assertEqual(base_path, "mlruns", f"MLflow tracking URI should point to 'mlruns', but it points to {tracking_uri}")

# Run the test
if __name__ == "__main__":
    unittest.main()