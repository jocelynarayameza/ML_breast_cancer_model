# Usa una imagen base de Python ligera (Requisito 3)
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia solo el archivo de requisitos primero para aprovechar el cache de Docker
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de los archivos al directorio de trabajo (model.pkl, api.py, etc.)
COPY . .

# Expone el puerto que usará Flask/Gunicorn
EXPOSE 5000

# Comando para correr la aplicación usando Gunicorn (servidor más robusto que el de desarrollo de Flask)
# Reemplaza 'api:app' con el nombre de tu archivo (api.py) y el objeto Flask (app)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api:app"]