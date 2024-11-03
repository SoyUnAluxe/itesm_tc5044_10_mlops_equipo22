
import unittest
import os
import pandas as pd
from src.data.preprocess import DataPreprocessor
import numpy as np

class TestDataPreprocessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Paths based on user's existing dataset
        cls.raw_data_path = "data/raw/Steel_industry_data.csv"
        cls.processed_data_path = "data/processed/processed_data.csv"

    def test_load_data(self):
        processor = DataPreprocessor(self.raw_data_path, self.processed_data_path)
        data = processor.load_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(data.shape[0], 0)  # Ensure data has rows
        self.assertGreater(data.shape[1], 0)  # Ensure data has columns

    def test_date_and_month_transformation(self):
        processor = DataPreprocessor(self.raw_data_path, self.processed_data_path)
        processor.load_data()
        X, y = processor.preprocess()

        # Check if 'month' column exists and is properly one-hot encoded in X
        month_columns = [col for col in X.columns if col.startswith("month_")]
        self.assertGreater(len(month_columns), 0)
        valid_months = ["August", "December", "February", "January", "July", 
                        "June", "March", "May", "November", "October", "September"]
        self.assertTrue(all(month.replace("month_", "") in valid_months for month in month_columns))

    def test_one_hot_encoding_days_of_week(self):
        processor = DataPreprocessor(self.raw_data_path, self.processed_data_path)
        processor.load_data()
        X, _ = processor.preprocess()

        # Check for expected one-hot encoded day columns in X
        day_columns = ["Day_of_week_Monday", "Day_of_week_Saturday", "Day_of_week_Sunday", 
                       "Day_of_week_Thursday", "Day_of_week_Tuesday", "Day_of_week_Wednesday"]
        for day in day_columns:
            self.assertIn(day, X.columns)

    def test_numeric_scaling(self):
        processor = DataPreprocessor(self.raw_data_path, self.processed_data_path)
        processor.load_data()
        X, _ = processor.preprocess()

        # Ensure numeric columns are scaled
        numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        for feature in numeric_features:
            if feature not in ['month', 'Load_Type']:  # exclude one-hot and categorical
                self.assertTrue(abs(X[feature].mean()) < 1, f"{feature} not scaled correctly")

    def test_target_encoding(self):
        processor = DataPreprocessor(self.raw_data_path, self.processed_data_path)
        processor.load_data()
        _, y = processor.preprocess()

        # Verify y contains encoded target variable 'Load_Type' with integer values, regardless of exact type
        self.assertIsInstance(y, np.ndarray)
        self.assertTrue(np.issubdtype(y.dtype, np.integer), "Target variable is not integer-encoded")


    def test_pca_components(self):
        processor = DataPreprocessor(self.raw_data_path, self.processed_data_path)
        processor.load_data()
        X, _ = processor.preprocess()

        # Check that PCA was applied and reduced columns exist
        pca_columns = [col for col in X.columns if col.startswith("PC")]
        self.assertEqual(len(pca_columns), 4)  # Expecting 4 components (PC1 to PC4)

if __name__ == "__main__":
    unittest.main()
