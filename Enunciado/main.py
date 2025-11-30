# run_merge_csv.py

from Utils import unir_csv_en_carpeta

if __name__ == "__main__":
    carpeta = "../BattleCity/DataCSV"            # carpeta donde están tus CSV
    salida = "./csv_unido.csv"       # archivo de salida

    unir_csv_en_carpeta(carpeta, salida)