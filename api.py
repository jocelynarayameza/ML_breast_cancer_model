import joblib
import pandas as pd
from flask import Flask, request, jsonify
import logging
import sys

# Configuración de logging (Requisito 6 Mínimo)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

# 1. Cargar el Modelo (Fuera del endpoint para rendimiento)
MODEL_FILE = 'model.pkl'
try:
    model = joblib.load(MODEL_FILE)
    logging.info(f"Modelo {MODEL_FILE} cargado correctamente.")
except FileNotFoundError:
    logging.error(f"Error: El archivo del modelo '{MODEL_FILE}' no se encontró. Ejecute 'python train.py' primero.")
    sys.exit(1)
except Exception as e:
    logging.error(f"Error al cargar el modelo: {e}")
    sys.exit(1)

# Inicializar Flask
app = Flask(__name__)

# Definir las 10 características que el modelo espera
EXPECTED_FEATURES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension'
]

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para recibir datos y retornar una predicción."""
    
    # 2. Manejo de errores de entrada
    if not request.json:
        logging.warning("Solicitud sin JSON o con formato incorrecto.")
        return jsonify({"error": "Debe enviar los datos en formato JSON."}), 400
    
    data = request.json
    
    # Validar que todas las características esperadas estén presentes
    try:
        input_data = {feature: data[feature] for feature in EXPECTED_FEATURES}
    except KeyError as e:
        logging.warning(f"Falta una característica requerida en el JSON: {e}")
        return jsonify({"error": f"Falta una o más características. Se esperan: {EXPECTED_FEATURES}"}), 400

    # 3. Preparación de los datos para la predicción
    try:
        input_df = pd.DataFrame([input_data])
        
        # 4. Hacer la Predicción
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0].tolist() # Convertir a lista para JSON
        
        # Convertir la predicción a tipo nativo de Python
        result = int(prediction)
        
        # Logging de la predicción (Requisito 6 Mínimo)
        logging.info(f"Predicción realizada: {result}. Probabilidades: {prediction_proba}")

        # 5. Retornar la respuesta (Requisito 2)
        return jsonify({
            "prediction": result,
            "probabilities": prediction_proba,
            "message": "Maligno" if result == 0 else "Benigno"
        })
    except Exception as e:
        # 6. Manejo de cualquier error inesperado
        logging.error(f"Error interno durante la predicción: {e}")
        return jsonify({"error": f"Ocurrió un error inesperado en el servidor: {e}"}), 500

# El servidor Gunicorn se encargará de ejecutar la aplicación en el despliegue.
# Para pruebas locales puedes usar: if __name__ == '__main__': app.run(debug=True, host='0.0.0.0', port=5000)