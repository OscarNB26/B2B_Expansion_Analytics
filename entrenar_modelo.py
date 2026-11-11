import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# 1. Generar datos ficticios de 1000 candidatos históricos
np.random.seed(42)
n_candidatos = 1000

datos = {
    'experiencia_directiva_anios': np.random.randint(1, 20, n_candidatos),
    'capital_disponible_k': np.random.randint(10, 100, n_candidatos),
    'score_habilidades_comerciales': np.random.randint(1, 10, n_candidatos),
    'sector_retail': np.random.randint(0, 2, n_candidatos) # 1 si viene de retail, 0 si no
}

df = pd.DataFrame(datos)

# 2. Crear la variable objetivo (Éxito = 1, Fracaso = 0)
# Simulamos que a más experiencia, capital y habilidades, más probabilidad de éxito
probabilidad = (df['experiencia_directiva_anios'] * 0.2 + 
                df['capital_disponible_k'] * 0.05 + 
                df['score_habilidades_comerciales'] * 0.5)
# Normalizar y convertir a 0 o 1
df['exito_12_meses'] = np.where(probabilidad + np.random.normal(0, 2, n_candidatos) > 8, 1, 0)

# 3. Entrenar el modelo
X = df.drop('exito_12_meses', axis=1)
y = df['exito_12_meses']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# 4. Guardar el modelo entrenado en un archivo
with open('modelo_scoring.pkl', 'wb') as archivo:
    pickle.dump(modelo, archivo)

print("Modelo entrenado y guardado correctamente.")