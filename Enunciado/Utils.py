# from skl2onnx import to_onnx
# from onnx2json import convert
import os
import pandas as pd
import pickle
import json


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