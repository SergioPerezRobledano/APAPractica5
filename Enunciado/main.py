# run_merge_csv.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score

from Utils import unir_csv_en_carpeta, cargar_y_preprocesar_csv, ExportAllformatsMLPSKlearn,WriteStandardScaler
from public_test import MLP_test_step_multi
#from MLP import MLP_backprop_predict
from MLPmulti import MLP_backprop_predict_multi

if __name__ == "__main__":
    carpeta = "../BattleCity/DataCSV"  # carpeta donde estan los CSV
    salida = "./csv_unido.csv"         # nombre del archivo de salida

    # 1 Unir CSVs
    unir_csv_en_carpeta(carpeta, salida)

    # 2 Cargar y preprocesar
    X_clean, Y,mean,stds = cargar_y_preprocesar_csv(salida)
    Y = Y.to_numpy()  # asegurarse de que sea NumPy

    # 3 Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, Y, test_size=0.33, random_state=0
    )

    # 4 One-hot encoding de Y
    encoder = OneHotEncoder(sparse_output=False)  # compatible con scikit-learn >=1.2
    y_train_encoded = encoder.fit_transform(y_train.reshape(-1,1))
    y_test_encoded  = encoder.transform(y_test.reshape(-1,1))


    # 4.5 Crear parametros 
    hidden_layers_global = [30,15]
    alpha_global = 0.0
    num_ite_global = 3500
    lambda_global = 0.92606
    verbose_global = 200

    # # 5 Ejecutar test con tu MLP personalizado
    # print("Test 1: Calculando para lambda = 0")
    # MLP_test_step_multi(
    #     MLP_backprop_predict_multi,
    #     1,
    #     X_train,
    #     y_train_encoded,
    #     X_test,
    #     y_test_encoded,
    #     alpha_global,        # alpha
    #     num_ite_global,    # num_ite
    #     lambda_global,     # lambda_
    #     verbose_global,    # verbose para mostrar las iteracciones
    #     hidden_layers=hidden_layers_global 
    # )

    # 6 Ejecutar test con MLP de SKLearn
    mlp_sk = MLPClassifier(hidden_layer_sizes=hidden_layers_global,
                           activation="logistic",
                           solver="sgd",
                           max_iter = num_ite_global,
                           learning_rate='constant', 
                           learning_rate_init = lambda_global,
                           n_iter_no_change = num_ite_global, 
                           alpha = alpha_global,
                           verbose=True,
                           random_state=0,
                           )

    mlp_sk.fit(X_train,y_train_encoded) # Para entrenar a la red con los dato s que le pasamos 
    acc = mlp_sk.score(X_test,y_test_encoded) # Esto saca la precision del modelo con respecto a los datos supuestos del test

    if(acc > 0.80):
        print("Guay del paraguay: " + str(acc))
    else: 
        print("MALMALMAL: " + str(acc))

    ExportAllformatsMLPSKlearn(mlp_sk, X_train,"Picklename", "onix", "json","Custom")
    WriteStandardScaler("./ExportarUnity/"+"StandarScaler",mean,stds)

    # # 8 Ejecutar test con MLP de SKLearn con parametros cambiados, usando relu y lbfgs
    # mlp_sk2 = MLPClassifier(hidden_layer_sizes=[50,40,35],
    #                        activation="relu",
    #                        solver="lbfgs",
    #                        max_iter = num_ite_global,
    #                        learning_rate="constant", 
    #                        learning_rate_init = lambda_global,
    #                        n_iter_no_change = num_ite_global, 
    #                        alpha = alpha_global,
    #                        verbose=True,
    #                        random_state=0,
    #                        )

    # mlp_sk2.fit(X_train,y_train_encoded) # Para entrenar a la red con los dato s que le pasamos 
    # acc2 = mlp_sk2.score(X_test,y_test_encoded) # Esto saca la precision del modelo con respecto a los datos supuestos del test

    # if(acc2 > 0.80):
    #     print("Guay del paraguay la segunda: " + str(acc2))
    # else: 
    #     print("MALMALMAL la segunda: " + str(acc2))

    # 9 Ejecutar un KNN para comprobar que tal va (entre 3 y 20 va de locos)
    # kneighbors = 7

    # knn_sk = KNeighborsClassifier(
    # n_neighbors=kneighbors,
    
    # # Esto usa ponderacion por distancia (el peso que tienen entre ellos por distancia)
    # weights='distance', 
    
    # metric='euclidean' 
# )

    # knn_sk.fit(X_train, y_train) # Entrenar con datos, no usar y_encoded porque si no sale con formato Multilabel-indicator y la precision usa una Multiclase (formato de lo que le pasas)

    # y_pred_knn = knn_sk.predict(X_test) # Prediccion
    # acc_knn = accuracy_score(y_test, y_pred_knn) # Precision de los datos

    # if(acc_knn > 0.80):
    #         print("Guay del paraguay la KNN: " + str(acc_knn))
    # else: 
    #         print("MALMALMAL la KNN: " + str(acc_knn))