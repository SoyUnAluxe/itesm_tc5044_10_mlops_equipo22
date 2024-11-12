# Usa una imagen de Python
FROM python:3.11.10-slim

# Configura el directorio de trabajo
WORKDIR /app

# Copia el archivo de requisitos y la estructura del proyecto
COPY requirements.txt requirements.txt

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código de la aplicación
COPY . .

# Expone el puerto 5000
EXPOSE 5000

# Ejecuta la aplicación Flask
CMD ["python", "app.py"]
