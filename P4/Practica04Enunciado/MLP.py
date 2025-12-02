# MLP.py
# Implementacion alternativa de una red neuronal de una capa oculta (MLP)
# Comentarios en espanol sin tildes y con mayor descripcion.

import numpy as np

class MLP:
    """
    Clase MLP: red neuronal con una capa oculta.
    Constructor crea los pesos iniciales theta1 y theta2 con distribucion uniforme.
    Parametros:
        inputLayer  : numero de unidades de entrada (sin contar bias)
        hiddenLayer : numero de unidades en la capa oculta
        outputLayer : numero de unidades de salida (numero de clases)
        seed        : semilla para reproducibilidad
        epislom     : rango para inicializacion uniforme [-epislom, epislom]
    """

    def __init__(self, inputLayer, hiddenLayer, outputLayer, seed=0, epislom=0.12):
        np.random.seed(seed)
        self.inputLayer = inputLayer
        self.hiddenLayer = hiddenLayer
        self.outputLayer = outputLayer

        # Peso de la capa 1: hidden x (input + 1) [incluye bias]
        self.theta1 = np.random.uniform(-epislom, epislom, (hiddenLayer, inputLayer + 1))
        # Peso de la capa 2: output x (hidden + 1) [incluye bias]
        self.theta2 = np.random.uniform(-epislom, epislom, (outputLayer, hiddenLayer + 1))

    # Permite cargar manualmente pesos ya entrenados (manteniendo forma)
    def new_trained(self, theta1, theta2):
        self.theta1 = theta1
        self.theta2 = theta2

    # Numero de ejemplos (metodo privado)
    def _m(self, x):
        return x.shape[0]

    # Funcion sigmoide (activacion)
    def _sigmoid(self, z):
        # Numericamente estable para evitar overflow
        return 1.0 / (1.0 + np.exp(-z))

    # Derivada de la sigmoide respecto a la activacion a
    def _sigmoid_prime(self, a):
        return a * (1.0 - a)

    # Feedforward: calcula activaciones y señales z
    def feedforward(self, x):
        m = self._m(x)
        # Activacion de entrada, con bias (columna de unos)
        a1 = np.concatenate([np.ones((m, 1)), x], axis=1)

        # Capa oculta: z2 = a1 * theta1^T, luego activacion a2; añadir bias
        z2 = a1.dot(self.theta1.T)
        a2_no_bias = self._sigmoid(z2)
        a2 = np.concatenate([np.ones((m, 1)), a2_no_bias], axis=1)

        # Capa de salida: z3 = a2 * theta2^T, luego activacion a3
        z3 = a2.dot(self.theta2.T)
        a3 = self._sigmoid(z3)

        # Devolvemos activaciones completas y valores z (sin aplicar activacion)
        return a1, a2, a3, z2, z3

    # Calculo del coste (cross-entropy) con regularizacion L2
    def compute_cost(self, yPrime, y, lambda_):
        m = self._m(y)
        eps = 1e-10  # para evitar log(0)
        # coste sin regularizacion
        term = - (y * np.log(yPrime + eps) + (1 - y) * np.log(1 - yPrime + eps))
        Jbase = np.sum(term) / m

        # regularizacion L2 (no se penaliza columna de bias)
        reg = (lambda_ / (2.0 * m)) * (np.sum(self.theta1[:, 1:] ** 2) + np.sum(self.theta2[:, 1:] ** 2))

        return Jbase + reg

    # Devuelve la clase con mayor activacion (argmax por fila)
    def predict(self, a3):
        p = np.argmax(a3, axis=1)
        return p

    # Calcula gradientes por backpropagation y coste J
    def compute_gradients(self, x, y, lambda_):
        m = self._m(x)
        # Forward
        a1, a2, a3, z2, z3 = self.feedforward(x)

        # Coste
        J = self.compute_cost(a3, y, lambda_)

        # Delta en salida
        delta3 = a3 - y  # m x outputLayer

        # Delta en capa oculta (sin bias)
        # theta2[:,1:] shape = (output, hidden)
        delta2_no_bias = delta3.dot(self.theta2[:, 1:]) * self._sigmoid_prime(a2[:, 1:])  # m x hidden

        # Gradientes acumulados (Delta)
        Delta1 = delta2_no_bias.T.dot(a1)  # hidden x (input+1)
        Delta2 = delta3.T.dot(a2)          # output x (hidden+1)

        # Normalizar por m
        grad1 = Delta1 / m
        grad2 = Delta2 / m

        # Regularizar columnas que no corresponden a bias
        grad1[:, 1:] += (lambda_ / m) * self.theta1[:, 1:]
        grad2[:, 1:] += (lambda_ / m) * self.theta2[:, 1:]

        return J, grad1, grad2

    # Gradiente de la regularizacion L2 para un theta dado (sin penalizar bias)
    def _regularizationL2Gradient(self, theta, lambda_, m):
        grad = np.zeros_like(theta)
        grad[:, 1:] = (lambda_ / m) * theta[:, 1:]
        return grad

    # Coste de la regularizacion L2
    def _regularizationL2Cost(self, m, lambda_):
        sumsq = np.sum(self.theta1[:, 1:] ** 2) + np.sum(self.theta2[:, 1:] ** 2)
        return (lambda_ / (2.0 * m)) * sumsq

    # Descenso por gradiente (backpropagation) para entrenar pesos
    def backpropagation(self, x, y, alpha, lambda_, numIte, verbose=0):
        """
        Entrena con descenso por gradiente:
            x, y    : datos
            alpha   : learning rate
            lambda_ : regularizacion L2
            numIte  : numero de iteraciones
            verbose : si >0, imprime cada 'verbose' iteraciones
        Devuelve historial de costes (lista de floats)
        """
        Jhistory = []
        for i in range(numIte):
            J, grad1, grad2 = self.compute_gradients(x, y, lambda_)
            # actualizar pesos
            self.theta1 = self.theta1 - alpha * grad1
            self.theta2 = self.theta2 - alpha * grad2

            # guardar historial
            Jhistory.append(J)

            # imprimir progreso si se solicita
            if verbose > 0:
                if (i % verbose == 0) or (i == numIte - 1):
                    print(f"Iteration {i+1:6}: Cost {J:8.6f}")

        return Jhistory


# Funciones auxiliares para pruebas publicas
def target_gradient(input_layer_size, hidden_layer_size, num_labels, x, y, reg_param):
    """
    Construye un MLP con las dimensiones dadas y calcula J y gradientes.
    Devuelve J, grad1, grad2, Theta1, Theta2
    """
    mlp = MLP(input_layer_size, hidden_layer_size, num_labels)
    J, grad1, grad2 = mlp.compute_gradients(x, y, reg_param)
    return J, grad1, grad2, mlp.theta1, mlp.theta2


def costNN(Theta1, Theta2, x, ys, reg_param):
    """
    Evalua el coste y gradientes dados Theta1 y Theta2 pasados externamente.
    Se crea una red temporal para evaluar compute_gradients.
    """
    # ojo: x.shape[1] es el numero de features
    mlp = MLP(x.shape[1], Theta1.shape[0], ys.shape[1])
    mlp.new_trained(Theta1, Theta2)
    J, grad1, grad2 = mlp.compute_gradients(x, ys, reg_param)
    return J, grad1, grad2


def MLP_backprop_predict(X_train, y_train, X_test, alpha, lambda_, num_ite, verbose):
    """
    Funcion wrapper usada en pruebas:
     - Crea una red con 25 unidades ocultas (igual que en original)
     - Entrena con backpropagation y devuelve predicciones para X_test
    """
    mlp = MLP(X_train.shape[1], 25, y_train.shape[1])
    Jhistory = mlp.backpropagation(X_train, y_train, alpha, lambda_, num_ite, verbose)
    a3 = mlp.feedforward(X_test)[2]
    y_pred = mlp.predict(a3)
    return y_pred
