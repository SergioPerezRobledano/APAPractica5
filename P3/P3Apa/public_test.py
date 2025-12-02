import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

"""
Unit test to check the cost computation.

Args:
    mlp (MLP): instancia de la red neuronal MLP.
    yprime (array_like): salida generada por la red neuronal (todas las clases).
    y_one_hot_encoding (array_like): salida real codificada en formato one-hot.
"""
def compute_cost_test(mlp, yprime, y_one_hot_encoding):
    J = mlp.compute_cost(yprime, y_one_hot_encoding)
    assert np.isclose(J, 0.28762916516), f"Error: Cost must be 0.28762916516 for a perfect prediction but got {J}"
    print("\033[92mAll tests passed!")


"""
Unit test to check the prediction.

Args:
    p (array_like): prediccion generada por la red neuronal (clase).
    y (array_like): salida real del dataset.
    accuracyFunction (function): implementacion de la funcion accuracy del archivo utils.
"""
def predict_test(p, y, accuracyFunction):
    numDiff = 0
    for i in range(p.shape[0]):
        if p[i] != y[i]:
            numDiff += 1
    assert numDiff == 124, f"Case 1: predict_test: there are 124 wrong predictions but got {numDiff}"
    accuracy = accuracyFunction(p, y)
    assert np.isclose(accuracy, 0.9752), f"Case 2: accuracy must be 0.9752 for a perfect prediction but got {accuracy}"
    print("\033[92mAll tests passed!")


# ----------------------------------------------------------
# EJERCICIO 3: Matriz de confusion, precision, recall y F1
# ----------------------------------------------------------

"""
Funcion para evaluar el rendimiento del modelo al predecir el digito 0.

Se asume que la clase positiva es el digito 0, y las demas clases se tratan
como negativas. Se calculan la matriz de confusion, precision, recall y F1-score.

Args:
    y_true (array_like): etiquetas reales (vector de enteros).
    y_pred (array_like): etiquetas predichas por la red neuronal (vector de enteros).

Return:
    cm (array_like): matriz de confusion 2x2.
    precision (float): precision del modelo para la clase 0.
    recall (float): recall (sensibilidad) del modelo para la clase 0.
    f1 (float): F1-score del modelo para la clase 0.
"""
def confusion_metrics_test(y_true, y_pred):
    # Convertimos las etiquetas a binarias: clase positiva = 0
    y_true_bin = (y_true == 0).astype(int)
    y_pred_bin = (y_pred == 0).astype(int)

    # Calculamos la matriz de confusion
    cm = confusion_matrix(y_true_bin, y_pred_bin)

    # Calculamos las metricas
    precision = precision_score(y_true_bin, y_pred_bin)
    recall = recall_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin)

    # Mostramos resultados
    print("\nMatriz de confusion (clase 0 vs resto):\n", cm)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    return cm, precision, recall, f1
