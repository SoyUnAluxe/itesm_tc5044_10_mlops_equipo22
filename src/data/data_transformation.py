import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def data_transformer(data):
    data['date'] = pd.to_datetime(data['date'], format="%d/%m/%Y %H:%M")
    data['month'] = data['date'].dt.month_name()

    numeric_variables = data.select_dtypes(include='number').columns.to_list()

    object_variables = data.select_dtypes(include = 'object').columns.to_list()

    object_variables.remove('Load_Type')

    #'Scaling' la data numérica
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[numeric_variables])

    #Aplicando reducción de dimensionalidad
    pca = PCA(n_components=0.95)  
    data_pca = pca.fit_transform(data_scaled)
    cantidad_variables = data_pca.shape[1]
    #print(f'Se redujo a {cantidad_variables} variables numéricas')
    pca_df = pd.DataFrame(data_pca, columns=[f'PC{i+1}' for i in range(data_pca.shape[1])])

    #Codificación de variables categóricas
    data_encoded_df = pd.get_dummies(data[object_variables], columns=object_variables, drop_first=True)
    #Concatenando nuevo dataframe
    X = pd.concat([data_encoded_df, pca_df], axis=1)
    #Variable objetivo codificada
    le = LabelEncoder()
    y = le.fit_transform(data['Load_Type'])
     

    
    return X, y
