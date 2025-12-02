import numpy as np
from matplotlib import pyplot
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder

"""
Muestra los datos 2D almacenados en X en una cuadricula visual.
"""
def displayData(X, example_width=None, figsize=(10, 10)):

    # Calcula filas y columnas
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Convierte a array 2D
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = n / example_width

    # Calcula el numero de imagenes a mostrar
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = pyplot.subplots(
        display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_width, example_width, order='F'),
                  cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')


"""
Carga los datos del dataset.
"""
def load_data(file):
    data = loadmat(file, squeeze_me=True)
    x = data['X']
    y = data['y']
    return x, y


"""
Carga los pesos desde el archivo de pesos.
"""
def load_weights(file):
    weights = loadmat(file)
    theta1, theta2 = weights['Theta1'], weights['Theta2']
    return theta1, theta2


"""
Implementacion del one-hot encoding.
Convierte las etiquetas numericas Y (por ejemplo, [1, 2, 3])
en vectores binarios (por ejemplo, [1,0,0], [0,1,0], [0,0,1]).

Usa OneHotEncoder de la libreria sklearn.
"""
def one_hot_encoding(Y):
    # Aseguramos que Y tenga forma de columna (m, 1)
    Y = Y.reshape(-1, 1)

    # Creamos el codificador y lo ajustamos a los datos
    encoder = OneHotEncoder(sparse_output=False)
    YEnc = encoder.fit_transform(Y)

    return YEnc


"""
Implementacion de la funcion de precision (accuracy).
Compara las predicciones P con las etiquetas reales Y.

Args:
    P (array_like): Vector de etiquetas predichas.
    Y (array_like): Vector de etiquetas reales.

Return:
    acc (float): Porcentaje de acierto del modelo.
"""
def accuracy(P, Y):
    # Calculamos el numero total de ejemplos
    m = len(Y)

    # Contamos cuantos ejemplos coinciden
    correct = np.sum(P == Y.flatten())

    # Calculamos el porcentaje de acierto
    acc = (correct / m) 

    return acc
