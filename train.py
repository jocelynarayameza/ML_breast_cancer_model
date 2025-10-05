import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Cargar el Dataset
print("Cargando el dataset Breast Cancer...")
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Usaremos solo las primeras 10 características para simplificar la API
features = data.feature_names[:10]
X = df[features]
y = df['target']

# 2. Separar datos (aunque no es estrictamente necesario para el requisito mínimo)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entrenar el Modelo
print("Entrenando modelo de Regresión Logística...")
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# 4. Evaluar (Mínimo)
score = model.score(X_test, y_test)
print(f"Precisión del modelo en el set de prueba: {score:.4f}")

# 5. Guardar el modelo entrenado (Requisito 1)
MODEL_FILE = 'model.pkl'
joblib.dump(model, MODEL_FILE)
print(f"Modelo guardado exitosamente como {MODEL_FILE}")

# Opcional: Imprimir las características esperadas para referencia
print("\nCaracterísticas esperadas en la API:")
print(list(features))