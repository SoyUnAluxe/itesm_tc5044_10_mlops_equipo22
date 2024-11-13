import mlflow
mlflow.set_tracking_uri("file:./mlruns")
import mlflow.sklearn
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
from datetime import datetime


class ModelTrainer:
    def __init__(self, X_path, y_path, config_path):
        self.X_path = X_path
        self.y_path = y_path
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"{self.config_path} not found.")
        with open(self.config_path, 'r') as conf_file:
            config = yaml.safe_load(conf_file)
        return config

    def load_data(self):
        if not os.path.exists(self.X_path) or not os.path.exists(self.y_path):
            raise FileNotFoundError(f"Data files {self.X_path} or {self.y_path} not found.")
        X = pd.read_csv(self.X_path)
        y = pd.read_csv(self.y_path).values.ravel()
        return X, y

    def plot_confusion_matrix(self, y_test, y_pred, model_name_unique):
        # Generar la matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Matriz de Confusión")
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.savefig(f"reports/confusion_matrix_{model_name_unique}.png")
        mlflow.log_artifact(f"reports/confusion_matrix_{model_name_unique}.png")


    def log_feature_importance(self, model, X_train, model_name_unique):
        # Solo aplicable para modelos que soportan feature_importances_
        if hasattr(model, 'feature_importances_'):
            feature_importances = pd.DataFrame(model.feature_importances_,
                                               index=X_train.columns,
                                               columns=['Importance']).sort_values('Importance', ascending=False)
            feature_importances.plot(kind='barh', figsize=(8,6))
            plt.title('Importancia de Características')
            plt.tight_layout()
            plt.savefig(f"reports/feature_importance_{model_name_unique}.png")
            mlflow.log_artifact(f"reports/feature_importance_{model_name_unique}.png")

    def train_and_log_model(self, model, model_name, params):
        # Establecer el experimento en MLflow
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

        # Iniciar una nueva corrida en MLflow
        with mlflow.start_run(run_name=model_name):
            # Cargar datos
            X, y = self.load_data()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=params['test_size'], 
                random_state=params.get('random_state', 42)
            )

            # Entrenar el modelo
            model.fit(X_train, y_train)

            # Hacer predicciones
            y_pred = model.predict(X_test)

            # Calcular métricas
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')

            # Registrar parámetros y métricas en MLflow
            mlflow.log_params(params)
            mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec})

            # Guardar el modelo en MLflow
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name_unique = f"{model_name}_{timestamp}"
            mlflow.sklearn.log_model(model, artifact_path=f"models/{model_name_unique}")

            # Opcional: Guardar el modelo localmente con un nombre único
            joblib.dump(model, f"saved_models/{model_name_unique}.pkl")

            # Graficar la matriz de confusión
            self.plot_confusion_matrix(y_test, y_pred, model_name_unique)

            # Registrar la importancia de características si aplica
            self.log_feature_importance(model, pd.DataFrame(X_train), model_name_unique)

            print(f"Modelo: {model_name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
            print(f"Modelo guardado como: {model_name_unique}")

    def run(self):
        # Leer la configuración de los modelos desde el archivo params.yaml
        models_config = self.config['models']

        for model_info in models_config:
            model_name = model_info['name']
            model_params = model_info['params']

            # Crear la instancia del modelo
            if model_name == "RandomForest":
                model = RandomForestClassifier(**{k: v for k, v in model_params.items() if k != 'test_size'})
            elif model_name == "KNeighbors":
                model = KNeighborsClassifier(**{k: v for k, v in model_params.items() if k != 'test_size'})
            elif model_name == "DecisionTree":
                model = DecisionTreeClassifier(**{k: v for k, v in model_params.items() if k != 'test_size'})
            else:
                raise ValueError(f"Modelo {model_name} no soportado")

            # Entrenar y registrar el modelo
            self.train_and_log_model(model, model_name, model_params)

if __name__ == "__main__":
    trainer = ModelTrainer(
        X_path=r'data/processed/processed_dataset_X.csv', 
        y_path=r'data/processed/processed_dataset_y.csv', 
        config_path='params.yaml'
    )
    trainer.run()
