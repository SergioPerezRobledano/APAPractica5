# utils.py
# Funciones utilitarias: carga de datos, codificacion one-hot, funciones para pruebas numericas,
# y display de imagenes.
# Comentarios en espanol sin tildes.

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

def displayData(X):
    """
    Muestra una cuadrricula de imagenes 20x20 a partir de X (cada fila es una imagen).
    Devuelve (fig, ax).
    """
    num_plots = int(np.sqrt(X.shape[0]))
    fig, ax = plt.subplots(num_plots, num_plots, sharex=True, sharey=True)
    plt.subplots_adjust(left=0, wspace=0, hspace=0)
    img_num = 0
    for i in range(num_plots):
        for j in range(num_plots):
            img = X[img_num, :].reshape(20, 20).T
            ax[i][j].imshow(img, cmap='Greys')
            ax[i][j].axis('off')
            img_num += 1
    return fig, ax

def displayImage(im):
    """
    Muestra una sola imagen 20x20 (im es un vector).
    """
    fig2, ax2 = plt.subplots()
    image = im.reshape(20, 20).T
    ax2.imshow(image, cmap='gray')
    return fig2, ax2

def load_data(file):
    """
    Carga datos desde un .mat (se asume que contiene X e y).
    Devuelve X, y.
    """
    data = loadmat(file, squeeze_me=True)
    X = data['X']
    y = data['y']
    return X, y

def load_weights(file):
    """
    Carga Theta1 y Theta2 desde un .mat.
    """
    weights = loadmat(file)
    theta1, theta2 = weights['Theta1'], weights['Theta2']
    return theta1, theta2

def one_hot_encoding(Y):
    """
    Codifica vector Y (n,) en matriz one-hot (n x k).
    Utiliza sklearn OneHotEncoder y devuelve matriz densa.
    """
    Y = Y.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    YEnc = encoder.fit_transform(Y)
    return YEnc

def accuracy(P, Y):
    """
    Calcula la exactitud simple (mean(P == Y)).
    Si Y esta en formato one-hot, se convierte a etiquetas.
    """
    if Y.ndim > 1:
        Y = np.argmax(Y, axis=1)
    return np.mean(P == Y)

# -----------------------
# Funciones para debug y gradiente numerico
# -----------------------

def debugInitializeWeights(fan_in, fan_out):
    """
    Inicializa pesos de forma determinista para pruebas:
    Genera valores sen(1..N)/10 y los organiza en forma Fortran (column major)
    igual que en la version original.
    """
    W = np.sin(np.arange(1, 1 + (1 + fan_in) * fan_out)) / 10.0
    W = W.reshape(fan_out, 1 + fan_in, order='F')
    return W

def computeNumericalGradient(J, Theta1, Theta2):
    """
    Calcula el gradiente numerico de una funcion J que acepta un vector
    con todos los parametros concatenados.
    La funcion J debe tener la forma J(theta_vector) -> coste (scalar).
    Theta1 y Theta2 se usan solo para obtener la longitud total y el reshape.
    """
    # concatenar Thetas en un vector
    theta = np.append(Theta1, Theta2).reshape(-1)
    numgrad = np.zeros_like(theta)
    perturb = np.zeros_like(theta)
    eps = 1e-4

    for p in range(len(theta)):
        perturb[p] = eps
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
        numgrad[p] = (loss2 - loss1) / (2.0 * eps)
        perturb[p] = 0.0

    return numgrad
