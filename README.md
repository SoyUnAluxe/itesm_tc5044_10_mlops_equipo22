
# API de Predicción con Modelo de Machine Learning

Este proyecto contiene una API en Flask para realizar predicciones utilizando un modelo de machine learning previamente entrenado. Sigue las instrucciones para generar el modelo, construir y ejecutar el contenedor Docker.

## Requisitos Previos

- Python 3.11.10
- Docker y Docker Compose
- DVC para gestionar el pipeline de datos

## Generación del Modelo

Si aún no tienes un modelo entrenado en el directorio `/saved_models`, sigue estos pasos para crear el pipeline de entrenamiento y generar un modelo.

### 1. Crear Directorios

Si no existen, crea los siguientes directorios en la raíz del proyecto:

```bash
mkdir -p reports saved_models
```

### 2. Instalar Dependencias

Instala las dependencias en un entorno virtual con Python 3.11.10:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Inicializar DVC

Ejecuta el siguiente comando para inicializar DVC:

```bash
dvc init
```

### 4. Ajustar Hiperparámetros

Modifica el archivo `params.yaml` para ajustar los hiperparámetros del modelo según tus necesidades.

### 5. Ejecutar el Pipeline de Entrenamiento

Corre el pipeline de DVC para entrenar los modelos y generar los archivos necesarios:

```bash
dvc repro
```

Esto generará tres modelos de machine learning en el directorio `/saved_models`, junto con los archivos `dummy_columns.pkl`, `label_encoder.pkl`, `pca.pkl` y `scaler.pkl`.

### 6. Seleccionar el Modelo

Copia el nombre del modelo de tu interés en el archivo `app.py` en la línea:

```python
model = joblib.load('saved_models/RandomForest_20241112_114024.pkl')
```

También puedes considerar agregar el nombre del modelo como un parámetro en `params.yaml` o en un nuevo archivo YAML para que sea más configurable.

## Construcción y Ejecución del Contenedor Docker

### 1. Construir el Contenedor

Ejecuta el siguiente comando para construir la imagen Docker:

```bash
docker build -t flask_model_api .
```

### 2. Ejecutar el Contenedor

Para ejecutar el contenedor, usa Docker Compose:

```bash
docker-compose up
```

Esto expondrá la API en `http://localhost:5000`.

## Realizar Peticiones

Puedes hacer una solicitud POST al endpoint `/predict` para obtener una predicción. A continuación, se muestra un ejemplo de solicitud:

### Ejemplo de Request

- **Método**: POST
- **URL**: `http://localhost:5000/predict`
- **Cuerpo** (JSON):

```json
{
    "date": "05/01/2018 13:15",
    "Usage_kWh": 50.08,
    "Lagging_Current_Reactive.Power_kVarh": 10.12,
    "Leading_Current_Reactive_Power_kVarh": 7.6,
    "CO2(tCO2)": 0.02,
    "Lagging_Current_Power_Factor": 98.02,
    "Leading_Current_Power_Factor": 100,
    "NSM": 900,
    "WeekStatus": "Weekday",
    "Day_of_week": "Monday"
}
```

La API responderá con una predicción en formato JSON, utilizando el modelo que configuraste.
