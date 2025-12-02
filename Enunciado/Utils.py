# from skl2onnx import to_onnx
# from onnx2json import convert
import os
import pandas as pd
import pickle
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# def ExportONNX_JSON_TO_Custom(onnx_json,mlp):
#     graphDic = onnx_json["graph"]
#     initializer = graphDic["initializer"]
#     s= "num_layers:"+str(mlp.n_layers_)+"\n"
#     index = 0
#     parameterIndex = 0;
#     for parameter in initializer:
#         name = parameter["name"]
#         print("Capa ",name)
#         if name != "classes" and name != "shape_tensor":
#             print("procesando ",name)
#             s += "parameter:"+str(parameterIndex)+"\n"
#             print(parameter["dims"])
#             s += "dims:"+str(parameter["dims"])+"\n"
#             print(parameter["name"])
#             s += "name:"+str(parameter["name"])+"\n"
#             print(parameter["doubleData"])
#             s += "values:"+str(parameter["doubleData"])+"\n"
#             index = index + 1
#             parameterIndex = index // 2
#         else:
#             print("Esta capa no es interesante ",name)
#     return s

# def ExportAllformatsMLPSKlearn(mlp,X,picklefileName,onixFileName,jsonFileName,customFileName):
#     with open(picklefileName,'wb') as f:
#         pickle.dump(mlp,f)
    
#     onx = to_onnx(mlp, X[:1])
#     with open(onixFileName, "wb") as f:
#         f.write(onx.SerializeToString())
    
#     onnx_json = convert(input_onnx_file_path=onixFileName,output_json_path=jsonFileName,json_indent=2)
    
#     customFormat = ExportONNX_JSON_TO_Custom(onnx_json,mlp)
#     with open(customFileName, 'w') as f:
#         f.write(customFormat)



def unir_csv_en_carpeta(carpeta_entrada, archivo_salida):
    """
    Lee todos los archivos .csv en una carpeta y los combina en un único csv.

    :param carpeta_entrada: Ruta de la carpeta donde están los .csv
    :param archivo_salida: Ruta del archivo .csv final
    """
    dataframes = []

    for archivo in os.listdir(carpeta_entrada):
        if archivo.endswith(".csv"):
            ruta = os.path.join(carpeta_entrada, archivo)
            print(f"Revisando: {ruta}")

            # Leer el archivo completo como texto para inspeccionar la última línea
            with open(ruta, "r", encoding="utf-8") as f:
                lineas = f.read().strip().split("\n")

            ultima_linea = lineas[-1].strip()

            # Verificar si la última línea empieza por "win"
            if ultima_linea.startswith("win"):
                print(f"✔ Archivo válido, se añade: {archivo}")

                # Leer CSV ignorando la última línea (que contiene win)
                df = pd.read_csv(ruta, on_bad_lines='skip')
                
                # Eliminar la fila "win" si entró
                df = df[df.iloc[:, 0] != "win"]

                dataframes.append(df)
            else:
                print(f"✖ Archivo ignorado (no termina en 'win'): {archivo}")

    # Si no hay archivos válidos, evitar error
    if not dataframes:
        print("No se encontró ningún CSV válido.")
        return

    df_final = pd.concat(dataframes, ignore_index=True)
    df_final.to_csv(archivo_salida, index=False)
    print(f"\nArchivo final guardado en: {archivo_salida}")



def cargar_y_preprocesar_csv(ruta_csv: str):
    # ============================================================
    # 1. Cargar CSV final
    # ============================================================
    df = pd.read_csv(ruta_csv)
    print("Tamaño inicial:", df.shape)

    # ============================================================
    # 2. LIMPIEZA DE DATOS
    # ============================================================

    # ---- 2.1 Eliminar filas con NaN
    df = df.dropna()
    print("Tras eliminar NaN:", df.shape)

    # ---- 2.2 Filtrar acciones inválidas
    df = df[df["action"].isin([0, 1, 2, 3, 4, 5])]
    print("Tras filtrar acciones inválidas:", df.shape)

    # ---- 2.3 Corregir posiciones imposibles
    def corregir_posicion(x):
        return x if 0 <= x <= 30 else 30

    pos_cols = [
        "COMMAND_CENTER_X","COMMAND_CENTER_Y",
        "AGENT_1_X","AGENT_1_Y",
        "AGENT_2_X","AGENT_2_Y",
        "LIFE_X","LIFE_Y",
        "EXIT_X","EXIT_Y"
    ]

    for col in pos_cols:
        df[col] = df[col].apply(corregir_posicion)

    # ---- 2.4 Sustituir agentes eliminados (100) por 30
    df[["AGENT_1_X","AGENT_1_Y"]] = df[["AGENT_1_X","AGENT_1_Y"]].replace(100, 30)
    df[["AGENT_2_X","AGENT_2_Y"]] = df[["AGENT_2_X","AGENT_2_Y"]].replace(100, 30)

    # ---- 2.5 Distancias negativas → absolutas
    dist_cols = [
        "NEIGHBORHOOD_DIST_UP","NEIGHBORHOOD_DIST_DOWN",
        "NEIGHBORHOOD_DIST_RIGHT","NEIGHBORHOOD_DIST_LEFT"
    ]

    for col in dist_cols:
        df[col] = df[col].apply(lambda x: abs(x))

    # ============================================================
    # 3. SEPARAR X e y
    # ============================================================
    X = df.drop("action", axis=1)
    y = df["action"]

    # ============================================================
    # 4. PREPROCESAMIENTO
    # ============================================================
    categorical_cols = [
        "NEIGHBORHOOD_UP",
        "NEIGHBORHOOD_DOWN",
        "NEIGHBORHOOD_RIGHT",
        "NEIGHBORHOOD_LEFT"
    ]

    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor)
    ])

    # Fit + transform
    X_clean = pipeline.fit_transform(X)

    print("Tamaño final:", X_clean.shape)
    print("Clases:", set(y))

    return X_clean, y