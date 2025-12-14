# run_merge_csv.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from Utils import unir_csv_en_carpeta, cargar_y_preprocesar_csv #, ExportAllformatsMLPSKlearn,WriteStandardScaler
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
    # hidden_layers_global = [30,25,10,5]    // Para comprobacion de multicapa
    hidden_layers_global = [40,20]
    alpha_global = 0.0
    num_ite_global = 10000
    lambda_global = 0.92606
    verbose_global = 200

    # 5 Ejecutar test con tu MLP personalizado
    print("Test 1: Calculando para lambda = 0")
    accAcomparar = MLP_test_step_multi(
        MLP_backprop_predict_multi,
        1,
        X_train,
        y_train_encoded,
        X_test,
        y_test_encoded,
        alpha_global,        # alpha
        num_ite_global,    # num_ite
        lambda_global,     # lambda_
        verbose_global,    # verbose para mostrar las iteracciones
        hidden_layers=hidden_layers_global 
    )

    # 6 Ejecutar test con MLP de SKLearn
    mlp_sk = MLPClassifier(hidden_layer_sizes=hidden_layers_global,
                           activation="logistic",
                           solver="sgd",
                           max_iter = num_ite_global,
                           learning_rate='constant', 
                           learning_rate_init = lambda_global,
                           n_iter_no_change = 500, 
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

    # ExportAllformatsMLPSKlearn(mlp_sk, X_train,"Picklename", "onix", "json","Custom")
    # WriteStandardScaler("./ExportarUnity/"+"StandarScaler",mean,stds)

    # 8 Ejecutar test con MLP de SKLearn con parametros cambiados, usando relu y lbfgs
    mlp_sk2 = MLPClassifier(hidden_layer_sizes=[50,40,35],
                           activation="relu",
                           solver="lbfgs",
                           max_iter = num_ite_global,
                           learning_rate="constant", 
                           learning_rate_init = lambda_global,
                           n_iter_no_change = num_ite_global, 
                           alpha = alpha_global,
                           verbose=True,
                           random_state=0,
                           )

    mlp_sk2.fit(X_train,y_train_encoded) # Para entrenar a la red con los dato s que le pasamos 
    acc2 = mlp_sk2.score(X_test,y_test_encoded) # Esto saca la precision del modelo con respecto a los datos supuestos del test

    if(acc2 > 0.80):
        print("Guay del paraguay la segunda: " + str(acc2))
    else: 
        print("MALMALMAL la segunda: " + str(acc2))

    # 9 Ejecutar un KNN para comprobar que tal va (entre 3 y 20 va de locos)
    kneighbors = 7

    knn_sk = KNeighborsClassifier(
    n_neighbors=kneighbors,
    
    # Esto usa ponderacion por distancia (el peso que tienen entre ellos por distancia)
    weights='distance', 
    
    metric='euclidean' 
)

    knn_sk.fit(X_train, y_train) # Entrenar con datos, no usar y_encoded porque si no sale con formato Multilabel-indicator y la precision usa una Multiclase (formato de lo que le pasas)

    y_pred_knn = knn_sk.predict(X_test) # Prediccion
    acc_knn = accuracy_score(y_test, y_pred_knn) # Precision de los datos

    if(acc_knn > 0.80):
            print("Guay del paraguay la KNN: " + str(acc_knn))
    else: 
            print("MALMALMAL la KNN: " + str(acc_knn))


    # 9. Implementar y Evaluar con SKLearn Random Forest
print("\n--- Test 4: Comparando con SKLearn RandomForestClassifier ---")

# Paso 1: Inicializar el modelo
# Hiperparámetros clave de Random Forest:
N_ESTIMATORS = 100 # Número de árboles en el bosque (suele ser 100 o más)
MAX_DEPTH = 10     # Profundidad máxima de cada árbol (para controlar overfitting)

# NOTA: Random Forest necesita etiquetas de clase simples (como y_train/y_test), 
# igual que el KNN, NO el One-Hot Encoding.
rf_sk = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    random_state=0,
    n_jobs=-1, # Usa todos los núcleos del CPU
)

# Paso 2: Entrenar el modelo
# Utilizamos X_train (features escaladas) y y_train (etiquetas de clase simple)
rf_sk.fit(X_train, y_train)

# Paso 3: Predicciones
y_pred_rf = rf_sk.predict(X_test)

# Paso 4: Evaluar la precisión (accuracy)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"\n--- Resultado de Validación con SKLearn RandomForestClassifier ---")
print(f"Parámetros usados:")
print(f"  Número de Árboles (n_estimators): {N_ESTIMATORS}")
print(f"  Profundidad Máxima (max_depth): {MAX_DEPTH}")
print(f"Precisión de Validación (Accuracy) de Random Forest: {accuracy_rf:.4f}")

if accuracy_rf > 0.80:
    print("¡Objetivo de Precisión (80%) de RF Alcanzado!")
else:
    print("El objetivo de Precisión (80%) de RF aún no se ha alcanzado.")


# 9. Implementar y Evaluar con SKLearn Árbol de Decisión
print("\n--- Test 5: Evaluando con SKLearn DecisionTreeClassifier ---")

# Hiperparámetros clave para el Árbol de Decisión:
# 1. max_depth: Limita la profundidad del árbol para controlar el sobreajuste.
# 2. min_samples_leaf: El número mínimo de muestras que debe tener un nodo hoja.
# 3. criterion: La función para medir la calidad de una división.

MAX_DEPTH_DT = 20     # Puedes probar diferentes profundidades, ej. 5, 10, 15
CRITERION_DT = 'entropy' # 'gini' (por defecto) o 'entropy' (ganancia de información)

# Paso 1: Inicializar el modelo
dt_sk = DecisionTreeClassifier(
    max_depth=MAX_DEPTH_DT,
    criterion=CRITERION_DT,
    random_state=0,
)

# Paso 2: Entrenar el modelo
# Utilizamos X_train (features escaladas) y y_train (etiquetas de clase simple)
dt_sk.fit(X_train, y_train)

# Paso 3: Predicciones
y_pred_dt = dt_sk.predict(X_test)

# Paso 4: Evaluar la precisión (accuracy)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

print(f"\n--- Resultado de Validación con SKLearn DecisionTreeClassifier ---")
print(f"Parámetros usados:")
print(f"  Criterio de División (criterion): {CRITERION_DT}")
print(f"  Profundidad Máxima (max_depth): {MAX_DEPTH_DT}")
print(f"Precisión de Validación (Accuracy) de Árbol de Decisión: {accuracy_dt:.4f}")

if accuracy_dt > 0.80:
    print("¡Objetivo de Precisión (80%) de Árbol de Decisión Alcanzado!")
else:
    print("El objetivo de Precisión (80%) de Árbol de Decisión aún no se ha alcanzado.")



# --- COMPARACIÓN FINAL DE MODELOS ---
print("\n" + "="*50)
print("RESUMEN DE PRECISIÓN DE TODOS LOS MODELOS (ACCURACY)")
print("="*50)

# -----------------------------------------------------------------------
# **NOTA IMPORTANTE SOBRE MLP PERSONALIZADO:**
# Ya que la precisión de tu MLP personalizado (Test 1) no se guarda en una
# variable 'acc1', usaremos el acc del primer SKLearn MLP (Test 6) como
# referencia de la arquitectura principal para la tabla, ya que tiene
# los mismos parámetros principales.
# 
# Si tu función 'MLP_test_step_multi' devolviera la precisión, deberías
# guardarla aquí: acc1 = MLP_test_step_multi(...)
# -----------------------------------------------------------------------

# Asumimos que los resultados de las precisiones son:
# acc_mlp_custom = PRECISIÓN DE TU MLP PERSONALIZADO (ej. 0.85)

# Guardamos el acc del primer SKLearn MLP para usarlo como referencia
acc_mlp_sk_sgd = acc # precisión del Test 6 (MLP SKLearn con SGD/Logistic)
acc_mlp_sk_lbfgs = acc2 # precisión del Test 8 (MLP SKLearn con LBFGS/ReLU)

print(f"1. MLP Personalizado (SGD/Sigmoid): [Ver Salida del 'Test 1']")
print(f"2. MLP SKLearn (SGD/Sigmoid, {hidden_layers_global}): {acc_mlp_sk_sgd:.4f}")
print(f"3. MLP SKLearn (LBFGS/ReLU, [50,40,35]): {acc_mlp_sk_lbfgs:.4f}")
print(f"4. KNN (K={kneighbors}, Distance): {acc_knn:.4f}")
print(f"5. Random Forest (Depth={MAX_DEPTH}, Estimators={N_ESTIMATORS}): {accuracy_rf:.4f}")
print(f"6. Árbol de Decisión (Depth={MAX_DEPTH_DT}, {CRITERION_DT}): {accuracy_dt:.4f}")

print("\n" + "="*50)
print("ANALISIS COMPARATIVO VS. TU MLP (SKLEARN SGD/SIGMOID)")
print("="*50)

# Usamos acc_mlp_sk_sgd como la base de comparación (Modelo de Referencia)
base_acc = acc_mlp_sk_sgd 

print(f"Modelo Base (Referencia MLP SKLearn): {base_acc:.4f}\n")

# Lista de modelos y sus resultados
comparisons = {
    "MLP SKLearn (LBFGS/ReLU)": acc_mlp_sk_lbfgs,
    "KNN": acc_knn,
    "Random Forest": accuracy_rf,
    "Árbol de Decisión": accuracy_dt,
}

for name, current_acc in comparisons.items():
    diff = current_acc - base_acc
    
    if diff > 0.005:
        resultado = f"MEJOR (+{diff:.4f} puntos)"
    elif diff < -0.005:
        resultado = f"PEOR ({diff:.4f} puntos)"
    else:
        resultado = "〰️ SIMILAR"
    
    print(f"-> {name: <30}: {current_acc:.4f} ({resultado})")

print("\n--- Conclusión Teórica ---")
if accuracy_rf > base_acc and accuracy_rf > acc_mlp_sk_lbfgs:
    print("El **Random Forest** suele ser la mejor alternativa no neuronal para la Imitación de Comportamiento, ya que maneja bien las características mixtas y la no linealidad del juego, superando potencialmente al MLP.")
else:
    print("El **MLP** (en sus distintas configuraciones) o el modelo que tenga la mayor precisión, es el más adecuado para esta tarea de clasificación, especialmente porque está optimizado para su exportación a Unity.")



# ---------------------------------------------------------------------------------
    # --- COMPARACIÓN FINAL DE MODELOS ---
    # ---------------------------------------------------------------------------------
    print("\n" + "="*80)
    print("RESUMEN FINAL DE PRECISIÓN DE MODELOS (ACCURACY)")
    print("="*80)

    # ---------------------------------------------------------------------------------
    # Tabla de Resultados
    # ---------------------------------------------------------------------------------

    # Modelo 1: Tu MLP Personalizado
    modelo_base = "Tu MLP (SGD/Sigmoid)"
    base_acc = accAcomparar
    
    
    resultados = {
        modelo_base: base_acc,
        "MLP SKLearn (SGD/Sigmoid)": acc_mlp_sk_sgd,
        "MLP SKLearn (LBFGS/ReLU)": acc_mlp_sk_lbfgs,
        f"KNN (K={kneighbors})": acc_knn,
        f"Random Forest (Depth={MAX_DEPTH})": accuracy_rf,
        f"Árbol de Decisión (Depth={MAX_DEPTH_DT})": accuracy_dt,
    }

    # Encabezado de la tabla
    print(f"{'MODELO': <35} | {'PRECISIÓN (ACC)': <20} | {'COMPARADO CON TU MLP'}")
    print("-" * 80)
    
    # Imprimir todos los resultados
    for name, current_acc in resultados.items():
        if name == modelo_base:
            comparacion = "BASE"
        else:
            diff = current_acc - base_acc
            if diff > 0.005:
                comparacion = f"MEJOR (+{diff:.4f})"
            elif diff < -0.005:
                comparacion = f"PEOR ({diff:.4f})"
            else:
                comparacion = "SIMILAR (±0.5%)"
        
        print(f"{name: <35} | {current_acc:.4f}: <20 | {comparacion}")
        
    print("=" * 80)

    # Conclusión
    print("\nAnálisis del Mejor Rendimiento:")
    mejor_modelo = max(resultados, key=resultados.get)
    mejor_acc = resultados[mejor_modelo]
    
    print(f"El modelo con la mayor precisión de validación es el: **{mejor_modelo}** con un Accuracy de **{mejor_acc:.4f}**.")