import numpy as np

class MLP:

    """
    Constructor: Inicializa el perceptron multicapa (MLP).

    Args:
        theta1 (array_like): Pesos para la primera capa de la red neuronal
                             (de la capa de entrada a la capa oculta).
        theta2 (array_like): Pesos para la segunda capa de la red neuronal
                             (de la capa oculta a la capa de salida).
    """
    def __init__(self, theta1, theta2):
        self.theta1 = theta1
        self.theta2 = theta2


    """
    Devuelve el numero de ejemplos (filas) en el conjunto de datos de entrada.

    Args:
        x (array_like): Datos de entrada (matriz de ejemplos).
    """
    def _size(self, x):
        return x.shape[0]
    

    """
    Calcula la funcion sigmoide de la entrada z.

    La funcion sigmoide se usa como funcion de activacion en las neuronas,
    para introducir no linealidad en el modelo.

    Args:
        z (array_like): Senal de activacion recibida por la capa.

    Return:
        array_like: Resultado de aplicar la funcion sigmoide elemento a elemento.
    """
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    """
    Ejecuta el paso de propagacion hacia adelante (feedforward) en la red neuronal.

    Este metodo calcula las activaciones de cada capa:
    - a1: activacion de la capa de entrada (con termino de sesgo anadido)
    - a2: activacion de la capa oculta (con termino de sesgo anadido)
    - a3: activacion de la capa de salida (salida final de la red)

    Tambien devuelve:
    - z2, z3: senales (entradas netas) antes de aplicar la funcion de activacion
              en las capas 2 y 3.

    Args:
        x (array_like): Datos de entrada (matriz de ejemplos, tamano m x 400).

    Return:
        a1, a2, a3, z2, z3 (array_like): Activaciones y senales intermedias.
    """
    def feedforward(self, x):
        # Numero de ejemplos de entrenamiento
        m = self._size(x)

        # ----- CAPA DE ENTRADA -----
        # Anadimos una columna de 1s a la matriz de entrada (bias)
        a1 = np.hstack([np.ones((m, 1)), x])

        # ----- CAPA OCULTA -----
        # Calculamos la senal neta (z2) multiplicando las entradas por los pesos
        z2 = np.dot(a1, self.theta1.T)

        # Aplicamos la funcion sigmoide para obtener las activaciones de la capa oculta
        a2 = self._sigmoid(z2)

        # Anadimos el termino de bias a la capa oculta
        a2 = np.hstack([np.ones((a2.shape[0], 1)), a2])

        # ----- CAPA DE SALIDA -----
        # Calculamos la senal neta (z3)
        z3 = np.dot(a2, self.theta2.T)

        # Aplicamos la funcion sigmoide para obtener la salida final de la red
        a3 = self._sigmoid(z3)

        # Devolvemos las activaciones y senales intermedias
        return a1, a2, a3, z2, z3


    """
    Calcula el coste (funcion de error) de las predicciones generadas por la red neuronal.

    La funcion utilizada es la entropia cruzada (cross-entropy), que mide
    la diferencia entre las predicciones y las etiquetas reales.

    Args:
        yPrime (array_like): Salida predicha por la red neuronal (a3).
        y (array_like): Etiquetas reales en formato one-hot (m x 10).

    Return:
        J (float): Valor del coste promedio.
    """
    def compute_cost(self, yPrime, y):
        m = y.shape[0]  # numero de ejemplos

        # Evitamos errores numericos con log(0)
        eps = 1e-8

        # Calculo del coste segun la formula de entropia cruzada
        J = (-1 / m) * np.sum(y * np.log(yPrime + eps) + (1 - y) * np.log(1 - yPrime + eps))
        return J


    """
    Obtiene la clase predicha (0 a 9) para cada ejemplo de entrada,
    basandose en la neurona con mayor activacion en la capa de salida.

    Args:
        a3 (array_like): Salida generada por la red neuronal (matriz m x 10).

    Return:
        p (array_like): Vector con la etiqueta predicha para cada ejemplo.
    """
    def predict(self, a3):
        # np.argmax devuelve el indice de la activacion mas alta (maxima probabilidad)
        p = np.argmax(a3, axis=1)
        return p
