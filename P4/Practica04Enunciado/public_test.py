# public_test.py
# Funciones de prueba y utilidades para verificar gradientes y accuracy.
# Comentarios en espanol sin tildes.

from utils import debugInitializeWeights, computeNumericalGradient
from sklearn.metrics import accuracy_score
import numpy as np

def checkNNGradients(costNN, target_gradient, reg_param=0):
    """
    Comprueba que la implementacion del gradiente sea correcta comparando
    el gradiente numerico con el calculado por backpropagation.
    Imprime la diferencia relativa.
    """
    # dimensiones pequeñas para el test
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # X inicial para pruebas (utiliza funcion proporcionada)
    X = debugInitializeWeights(input_layer_size - 1, m)

    # Construir vectores de salida ys en one-hot
    y = [(i % num_labels) for i in range(m)]
    ys = np.zeros((m, num_labels))
    for i in range(m):
        ys[i, y[i]] = 1

    # Obtener coste y gradientes via funcion target
    cost, grad1, grad2, Theta1, Theta2 = target_gradient(input_layer_size, hidden_layer_size, num_labels, X, ys, reg_param)

    # Aplanar gradientes en un vector
    grad = np.concatenate((np.ravel(grad1), np.ravel(grad2)))

    # Funcion reducida que acepta un vector p, reconstruye Thetas y devuelve coste
    def reduced_cost_func(p):
        Theta1_r = np.reshape(p[:hidden_layer_size * (input_layer_size + 1)],
                              (hidden_layer_size, (input_layer_size + 1)))
        Theta2_r = np.reshape(p[hidden_layer_size * (input_layer_size + 1):],
                              (num_labels, (hidden_layer_size + 1)))
        return costNN(Theta1_r, Theta2_r, X, ys, reg_param)[0]

    # Calcular gradiente numerico
    numgrad = computeNumericalGradient(reduced_cost_func, Theta1, Theta2)

    # Calcular diferencia relativa
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)

    print('Si tu implementacion de compute_gradient es correcta, la diferencia relativa sera pequeña (menor que 1e-9).')
    print('Relative Difference: %g' % diff)
    if diff < 1e-9:
        print("Test passed!")
    else:
        print("Error en la implementacion del gradiente.")


def MLP_test_step(mlp_backprop_predict, alpha, X_train, y_train, X_test, y_test, lambda_, num_ite, baseLineAccuracy, verbose=0):
    """
    Ejecuta un paso de test: entrena la red con los parametros y calcula accuracy.
    Imprime el resultado comparado con el baseline proporcionado.
    """
    y_pred = mlp_backprop_predict(X_train, y_train, X_test, alpha, lambda_, num_ite, verbose)
    accu = accuracy_score(y_test, y_pred)
    print(f"Calculate accuracy for lambda = {(lambda_):1.5f} : {(accu):1.5f} expected accuracy is aprox: {(baseLineAccuracy):1.5f}")
