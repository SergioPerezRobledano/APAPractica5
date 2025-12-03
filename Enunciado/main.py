# run_merge_csv.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from Utils import unir_csv_en_carpeta, cargar_y_preprocesar_csv
from public_test import MLP_test_step_multi
#from MLP import MLP_backprop_predict
from MLPmulti import MLP_backprop_predict_multi

if __name__ == "__main__":
    carpeta = "../BattleCity/DataCSV"  # carpeta donde estan los CSV
    salida = "./csv_unido.csv"         # nombre del archivo de salida

    # 1 Unir CSVs
    unir_csv_en_carpeta(carpeta, salida)

    # 2 Cargar y preprocesar
    X_clean, Y = cargar_y_preprocesar_csv(salida)
    Y = Y.to_numpy()  # asegurarse de que sea NumPy

    # 3 Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, Y, test_size=0.33, random_state=0
    )

    # 4 One-hot encoding de Y
    encoder = OneHotEncoder(sparse_output=False)  # compatible con scikit-learn >=1.2
    y_train_encoded = encoder.fit_transform(y_train.reshape(-1,1))
    y_test_encoded  = encoder.transform(y_test.reshape(-1,1))

    # 5 Ejecutar test con tu MLP personalizado
    print("Test 1: Calculando para lambda = 0")
    MLP_test_step_multi(
        MLP_backprop_predict_multi,
        1,
        X_train,
        y_train_encoded,
        X_test,
        y_test_encoded,
        0.0,      # alpha
        5000,   # num_ite
        0.92606,# lambda_
        200,    # verbose
        hidden_layers=[50, 30, 20]
    )
