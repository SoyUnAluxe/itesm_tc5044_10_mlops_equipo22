import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

class DataPreprocessor:
    def __init__(self, raw_data_path, processed_data_path):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.data = None

    def load_data(self):
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"{self.raw_data_path} not found.")
        self.data = pd.read_csv(self.raw_data_path)
        return self.data

    def preprocess(self):
        processed_data = self.data.copy()

        # Convertir la columna 'date' a datetime y crear una columna 'month'
        processed_data['date'] = pd.to_datetime(processed_data['date'], format="%d/%m/%Y %H:%M")
        processed_data['month'] = processed_data['date'].dt.month_name()

        # Variables numéricas y categóricas
        numeric_variables = processed_data.select_dtypes(include='number').columns.to_list()
        object_variables = processed_data.select_dtypes(include='object').columns.to_list()

        if 'Load_Type' in object_variables:
            object_variables.remove('Load_Type')

        # Escalar los datos numéricos
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(processed_data[numeric_variables])

        # Reducción de dimensionalidad usando PCA
        pca = PCA(n_components=0.95)
        data_pca = pca.fit_transform(data_scaled)
        pca_df = pd.DataFrame(data_pca, columns=[f'PC{i+1}' for i in range(data_pca.shape[1])])

        # Codificación de variables categóricas
        data_encoded_df = pd.get_dummies(processed_data[object_variables], columns=object_variables, drop_first=True)

        # Concatenación del dataframe codificado y el PCA
        X = pd.concat([data_encoded_df, pca_df], axis=1)

        # Codificación de la variable objetivo
        le = LabelEncoder()
        y = le.fit_transform(processed_data['Load_Type'])

        return X, y

    def save_data(self, X, y):
        os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
        X.to_csv(self.processed_data_path.replace('.csv', '_X.csv'), index=False)
        pd.DataFrame(y, columns=['Load_Type']).to_csv(self.processed_data_path.replace('.csv', '_y.csv'), index=False)

    def run(self):
        self.load_data()
        X, y = self.preprocess()
        self.save_data(X, y)

if __name__ == "__main__":
    preprocessor = DataPreprocessor(r'data/raw/Steel_industry_data.csv', r'data/processed/processed_dataset.csv')
    preprocessor.run()
