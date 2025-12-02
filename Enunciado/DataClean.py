import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ============================================================
# 1. Cargar CSV final
# ============================================================
df = pd.read_csv("csv_unido.csv")

print("Tamaño inicial:", df.shape)
print(df.head())

# ============================================================
# 2. LIMPIEZA DE DATOS
# ============================================================

# ---- 2.1 Eliminar filas que tengan NaN o valores completamente vacíos
df = df.dropna()
print("Tras eliminar NaN:", df.shape)

# ---- 2.2 Filtrar acciones inválidas (por si existiera ruido)
df = df[df["action"].isin([0,1,2,3,4,5])]  # ejemplo si solo existen estas acciones
print("Tras filtrar acciones inválidas:", df.shape)

# ---- 2.3 Corregir posiciones imposibles
# Posición válida en el mapa  0 y 30 
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

# ---- 2.4 Sustituir agentes eliminados (100) por NaN → luego imputar con 30 o media
df[["AGENT_1_X","AGENT_1_Y"]] = df[["AGENT_1_X","AGENT_1_Y"]].replace(100, 30)
df[["AGENT_2_X","AGENT_2_Y"]] = df[["AGENT_2_X","AGENT_2_Y"]].replace(100, 30)

# ---- 2.5 Comprobar distancias negativas o imposibles (NEIGHBORHOOD_DIST_*)
dist_cols = [
    "NEIGHBORHOOD_DIST_UP","NEIGHBORHOOD_DIST_DOWN",
    "NEIGHBORHOOD_DIST_RIGHT","NEIGHBORHOOD_DIST_LEFT"
]

for col in dist_cols:
    df[col] = df[col].apply(lambda x: abs(x))  # si hubiera negativas

# ============================================================
# 3. SEPARAR ATRIBUTOS Y CLASE
# ============================================================
X = df.drop("action", axis=1)
y = df["action"]

# ============================================================
# 4. PREPROCESAMIENTO:
#    - OneHotEncoding a NEIGHBORHOOD_XXX (son IDs)
#    - Normalizar todo lo numérico
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

# Usamos un pipeline para preparar los datos automáticamente:
pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor)
])

# Fit-transform
X_clean = pipeline.fit_transform(X)

print("Tamaño final del dataset procesado:", X_clean.shape)
print("Clases:", set(y))
