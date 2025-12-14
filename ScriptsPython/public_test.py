# public_test.py
# Funciones de prueba y utilidades para verificar gradientes y accuracy.
# Comentarios en espanol sin tildes.

from UPerceptron import debugInitializeWeights, computeNumericalGradient
from sklearn.metrics import accuracy_score
from MLPmulti import MLP_backprop_predict_multi
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
    y_test_classes = np.argmax(y_test, axis=1)
    accu = accuracy_score(y_test_classes, y_pred)
    print(f"Calculate accuracy for lambda = {(lambda_):1.5f} : {(accu):1.5f} expected accuracy is aprox: {(baseLineAccuracy):1.5f}")

def MLP_test_step_multi(MLP_backprop_predict_multi, alpha, X_train, y_train, X_test, y_test, lambda_, num_ite, baseLineAccuracy, verbose,hidden_layers):
    """
    Ejecuta un paso de test usando un MLP de N capas ocultas.
    Entrena la red con los parametros dados, calcula accuracy
    e imprime comparacion con el baseline.

    Parametros:
        mlp_backprop_predict : funcion wrapper para entrenamiento+prediccion
        alpha                : learning rate
        X_train, y_train     : datos de entrenamiento
        X_test, y_test       : datos de prueba codificados one-hot
        lambda_              : regularizacion
        num_ite              : iteraciones
        baseLineAccuracy     : precision esperada
        verbose              : frecuencia de impresion
        hidden_layers        : lista con sizes de capas ocultas
    """

    # entrenar y predecir usando las capas ocultas especificadas
    y_pred = MLP_backprop_predict_multi(
        X_train,
        y_train,
        X_test,
        alpha,
        lambda_,
        num_ite,
        verbose,
        hidden_layers
    )

    # convertir one-hot a clases
    y_test_classes = np.argmax(y_test, axis=1)

    # calcular accuracy
    accu = accuracy_score(y_test_classes, y_pred)

    print(
        f"Calculate accuracy for lambda = {lambda_:1.5f} : {accu:1.5f} "
        f"expected accuracy is aprox: {baseLineAccuracy:1.5f}"
    )
    return accu