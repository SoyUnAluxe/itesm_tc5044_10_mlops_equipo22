import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import joblib

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

    def preprocess(self, train = True, processed_data = None):
        if train:
            processed_data = self.data.copy()

        # Convertir la columna 'date' a datetime y crear una columna 'month'
        processed_data['date'] = pd.to_datetime(processed_data['date'], format="%d/%m/%Y %H:%M")
        processed_data['month'] = processed_data['date'].dt.month_name()

        # Variables numéricas y categóricas
    
        numeric_variables = ['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh', 'CO2(tCO2)', 'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'NSM']
        object_variables = ['WeekStatus', 'Day_of_week', 'Load_Type', 'month']

        if 'Load_Type' in object_variables:
            object_variables.remove('Load_Type')

        # Escalar los datos numéricos
        if not train:
            print("Loading Scaler")
            scaler = joblib.load("saved_models/scaler.pkl")
            data_scaled = scaler.transform(processed_data[numeric_variables])
        else:
            print("Training Scaler")
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(processed_data[numeric_variables])

        
        if train:
            joblib.dump(scaler, "saved_models/scaler.pkl")

        # Reducción de dimensionalidad usando PCA
        
        if not train:
            print("Loading PCA")
            pca = joblib.load("saved_models/pca.pkl")
            data_pca = pca.transform(data_scaled)
        else:
            print("Training PCA")
            pca = PCA(n_components=0.95)
            data_pca = pca.fit_transform(data_scaled)
        
        if train:
            joblib.dump(pca, "saved_models/pca.pkl")
        pca_df = pd.DataFrame(data_pca, columns=[f'PC{i+1}' for i in range(data_pca.shape[1])])

        # Codificación de variables categóricas
        if train:
            data_encoded_df = pd.get_dummies(processed_data[object_variables], columns=object_variables, drop_first=True)
        else:
            data_encoded_df = pd.get_dummies(processed_data[object_variables], columns=object_variables, drop_first=False)

        dummy_columns = ['WeekStatus_Weekend', 'Day_of_week_Monday', 'Day_of_week_Saturday',
                            'Day_of_week_Sunday', 'Day_of_week_Thursday', 'Day_of_week_Tuesday',
                            'Day_of_week_Wednesday', 'month_August', 'month_December',
                            'month_February', 'month_January', 'month_July', 'month_June',
                            'month_March', 'month_May', 'month_November', 'month_October',
                            'month_September']
        
        if not train:
            
            # Asegurarse de que los datos de entrada tengan las mismas columnas
            for col in dummy_columns:
                if col not in data_encoded_df.columns:
                    data_encoded_df[col] = False
        
        data_encoded_df = data_encoded_df[dummy_columns]


        # Concatenación del dataframe codificado y el PCA
        X = pd.concat([data_encoded_df, pca_df], axis=1)

        if train:
            # Codificación de la variable objetivo
            le = LabelEncoder()
            y = le.fit_transform(processed_data['Load_Type'])
            joblib.dump(le, "saved_models/label_encoder.pkl")
        else:
            y = None

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
