import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, load_weights, one_hot_encoding, accuracy
from MLP import MLP
from public_test import compute_cost_test, predict_test, confusion_metrics_test

# -----------------------------------------------------------
# PRACTICA 03 - Redes Neuronales - Feedforward ANN
# -----------------------------------------------------------

# 1. Cargamos los datos y los pesos entrenados
x, y = load_data('data/ex3data1.mat')
theta1, theta2 = load_weights('data/ex3weights.mat')

# 2. Creamos una instancia del modelo MLP
mlp = MLP(theta1, theta2)

# 3. Ejecutamos el paso de feedforward
a1, a2, a3, z2, z3 = mlp.feedforward(x)

# 4. Obtenemos las predicciones
p = mlp.predict(a3)

# -----------------------------------------------------------
# TEST 1 - Verificacion de la prediccion y precision (Ejercicio 1)
# -----------------------------------------------------------
print("\n--- Ejecutando test de prediccion ---")
predict_test(p, y, accuracy)

# -----------------------------------------------------------
# TEST 2 - Verificacion del coste (Ejercicio 2)
# -----------------------------------------------------------
print("\n--- Ejecutando test de coste ---")
y_one_hot = one_hot_encoding(y)
compute_cost_test(mlp, a3, y_one_hot)

# -----------------------------------------------------------
# TEST 3 - Evaluacion de la clase 0 (Ejercicio 3)
# -----------------------------------------------------------
print("\n--- Evaluacion del modelo para la clase 0 ---")
confusion_metrics_test(y, p)
