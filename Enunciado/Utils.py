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

    # Recorre todos los archivos de la carpeta
    for archivo in os.listdir(carpeta_entrada):
        if archivo.endswith(".csv"):
            ruta = os.path.join(carpeta_entrada, archivo)
            print(f"Leyendo: {ruta}")
            df = pd.read_csv(ruta)
            dataframes.append(df)

    # Une todos los dataframes
    df_final = pd.concat(dataframes, ignore_index=True)

    # Guarda el resultado
    df_final.to_csv(archivo_salida, index=False)
    print(f"Archivo final guardado en: {archivo_salida}")