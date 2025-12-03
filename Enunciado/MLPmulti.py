# MLP_N_Hidden.py
# Implementacion de una red neuronal MLP con N capas ocultas
# Version generalizada del script original para permitir multiples capas ocultas

import numpy as np

class MLPmulti:
    """
    Clase MLP: red neuronal con N capas ocultas.
    Parametros:
        inputLayer    : numero de unidades de entrada (sin bias)
        hiddenLayers  : lista con el numero de unidades por cada capa oculta
        outputLayer   : numero de unidades de salida
        seed          : semilla para reproducibilidad
        epislom       : rango de inicializacion uniforme [-epislom, epislom]

    Se generan los pesos como una lista self.Thetas donde:
        Thetas[0] conecta input -> hidden1
        Thetas[1] conecta hidden1 -> hidden2
        ...
        Thetas[-1] conecta ultima oculta -> salida
    """

    def __init__(self, inputLayer, hiddenLayers, outputLayer, seed=0, epislom=0.12):

        np.random.seed(seed)

        self.inputLayer = inputLayer
        self.hiddenLayers = hiddenLayers
        self.outputLayer = outputLayer

        # Construimos lista con tamaños de todas las capas (incluye entrada y salida)
        layers = [inputLayer] + hiddenLayers + [outputLayer]

        # Lista de matrices de pesos
        self.Thetas = []

        # Crear todos los pesos con bias incluido
        for i in range(len(layers)-1):
            fan_in = layers[i] + 1      # +1 por bias
            fan_out = layers[i+1]
            theta = np.random.uniform(-epislom, epislom, (fan_out, fan_in))
            self.Thetas.append(theta)

    # Metodo para cargar pesos ya entrenados
    def new_trained(self, list_of_thetas):
        self.Thetas = list_of_thetas

    # numero de ejemplos
    def _m(self, x):
        return x.shape[0]

    # Activacion
    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _sigmoid_prime(self, a):
        return a * (1 - a)

    # -------------------------------------------------------------------------
    # FEEDFORWARD GENERALIZADO
    # -------------------------------------------------------------------------
    def feedforward(self, x):

        x = np.atleast_2d(x)
        m = self._m(x)

        activations = []   # a^(l)
        zs = []            # z^(l)

        # capa de entrada con bias
        a = np.concatenate([np.ones((m, 1)), x], axis=1)
        activations.append(a)

        # recorre todas las capas ocultas + salida
        for theta in self.Thetas:
            z = activations[-1].dot(theta.T)
            zs.append(z)
            a_no_bias = self._sigmoid(z)

            # no agregamos bias a la ultima capa (la de salida)
            if theta is not self.Thetas[-1]:
                a = np.concatenate([np.ones((m, 1)), a_no_bias], axis=1)
            else:
                a = a_no_bias

            activations.append(a)

        return activations, zs

    # -------------------------------------------------------------------------
    # COSTE
    # -------------------------------------------------------------------------
    def compute_cost(self, yPred, y, lambda_):
        m = self._m(y)
        eps = 1e-10

        term = - (y * np.log(yPred + eps) + (1 - y) * np.log(1 - yPred + eps))
        Jbase = np.sum(term) / m

        # regularizacion
        reg = 0
        for theta in self.Thetas:
            reg += np.sum(theta[:, 1:] ** 2)

        reg = (lambda_ / (2.0 * m)) * reg

        return Jbase + reg

    def predict(self, a_last):
        return np.argmax(a_last, axis=1)

    # -------------------------------------------------------------------------
    # BACKPROP GENERALIZADO
    # -------------------------------------------------------------------------
    def compute_gradients(self, x, y, lambda_):

        m = self._m(x)

        activations, zs = self.feedforward(x)

        # a^(L) (ultima capa)
        a_last = activations[-1]

        # coste
        J = self.compute_cost(a_last, y, lambda_)

        # listas para deltas y gradientes
        deltas = [None] * len(self.Thetas)
        grads = [None] * len(self.Thetas)

        # delta de la capa de salida
        deltas[-1] = (a_last - y)  # m x out

        # backprop para capas ocultas
        for l in reversed(range(len(self.Thetas)-1)):
            theta_next = self.Thetas[l+1]
            delta_next = deltas[l+1]

            # removemos columna de bias del theta_next
            delta_no_bias = delta_next.dot(theta_next[:, 1:])

            a_l_plus_1 = activations[l+1][:, 1:]  # sin bias
            deltas[l] = delta_no_bias * self._sigmoid_prime(a_l_plus_1)

        # gradientes
        for l in range(len(self.Thetas)):
            a_l = activations[l]              # incluye bias
            delta_l_plus_1 = deltas[l]        # no incluye bias
            grads[l] = (delta_l_plus_1.T.dot(a_l)) / m

            # regularizacion excepto columna bias
            grads[l][:, 1:] += (lambda_ / m) * self.Thetas[l][:, 1:]

        return J, grads

    # -------------------------------------------------------------------------
    # ENTRENAMIENTO
    # -------------------------------------------------------------------------
    def backpropagation(self, x, y, alpha, lambda_, numIte, verbose=0):

        Jhistory = []

        for i in range(numIte):

            J, grads = self.compute_gradients(x, y, lambda_)

            # actualizacion
            for l in range(len(self.Thetas)):
                self.Thetas[l] -= alpha * grads[l]

            Jhistory.append(J)

            if verbose > 0:
                if (i % verbose == 0) or (i == numIte - 1):
                    print(f"Iteration {i+1:6}: Cost {J:8.6f}")

        return Jhistory

def MLP_backprop_predict_multi(X_train, y_train_encoded, X_test, alpha, lambda_, num_ite, verbose,
                         hidden_layers=[25]):
    """
    Funcion wrapper usada en pruebas:
    - Crea una red MLP con las capas ocultas especificadas en hidden_layers
    - Entrena con backpropagation y devuelve predicciones para X_test

    Parametros:
        X_train          : datos de entrenamiento
        y_train_encoded  : etiquetas one-hot
        X_test           : datos de prueba
        alpha            : learning rate
        lambda_          : regularizacion L2
        num_ite          : iteraciones
        verbose          : frec. de impresion
        hidden_layers    : lista con el numero de unidades ocultas por capa (default [25])
    """

    # numero de clases
    n_output = y_train_encoded.shape[1]

    # inicializar red MLP con N capas ocultas
    mlp = MLPmulti(
        inputLayer=X_train.shape[1],
        hiddenLayers=hidden_layers,
        outputLayer=n_output
    )

    # entrenamiento
    Jhistory = mlp.backpropagation(X_train, y_train_encoded, alpha, lambda_, num_ite, verbose)

    # forward en test
    activations, _ = mlp.feedforward(X_test)
    a_last = activations[-1]     # salida final

    # prediccion
    y_pred = mlp.predict(a_last)

    return y_pred
