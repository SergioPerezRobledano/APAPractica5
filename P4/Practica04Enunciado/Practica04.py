# main.py
# Script principal que ejecuta los test y compara con sklearn.
# Comentarios en espanol sin tildes.

from MLP import MLP, target_gradient, costNN, MLP_backprop_predict
from utils import load_data, load_weights, one_hot_encoding, accuracy
from public_test import checkNNGradients, MLP_test_step
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def gradientTest():
    """
    Ejecuta las comprobaciones de gradiente con lambda 0 y 1
    """
    checkNNGradients(costNN, target_gradient, 0)
    checkNNGradients(costNN, target_gradient, 1)

def MLP_test(X_train, y_train, X_test, y_test):
    """
    Ejecuta pruebas de rendimiento y compara con sklearn.MLPClassifier.
    Los parametros y supuestos se mantienen como en la version original.
    """

    print("Se asume: random_state de train_test_split = 0, alpha=1, num_iterations = 2000, test_size=0.33, seed=0 y epislom = 0.12")

    # Asegurar que y_test y y_train son vectores de etiquetas (no codificadas)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)

    print("Test 1 Calculando para lambda = 0")
    MLP_test_step(MLP_backprop_predict, 1, X_train, y_train, X_test, y_test, 0, 2000, 0.92606, 200)
    print("Test 2 Calculando para lambda = 0.5")
    MLP_test_step(MLP_backprop_predict, 1, X_train, y_train, X_test, y_test, 0.5, 2000, 0.92545, 200)
    print("Test 3 Calculando para lambda = 1")
    MLP_test_step(MLP_backprop_predict, 1, X_train, y_train, X_test, y_test, 1, 2000, 0.92667, 200)

    # Comparacion con sklearn
    print("Comparacion con sklearn.MLPClassifier")
    if y_train.ndim > 1:
        y_train = np.argmax(y_train, axis=1)

    clf = MLPClassifier(hidden_layer_sizes=(25,), activation='logistic',
                        alpha=1, learning_rate_init=1.0, max_iter=2000,
                        random_state=0, verbose=False)
    clf.fit(X_train, y_train)
    y_pred_sklearn = clf.predict(X_test)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"Accuracy sklearn.MLPClassifier: {acc_sklearn:.5f}")


def main():
    print("Programa principal")

    # Test 1: comprobacion de gradientes
    gradientTest()

    # Test 2: entrenamiento y evaluacion
    X, y = load_data("data/ex3data1.mat")
    y_encoded = one_hot_encoding(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.33, random_state=0)

    MLP_test(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
