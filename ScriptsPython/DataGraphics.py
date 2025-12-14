# ================================================
# 1. Importar librerías
# ================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# ================================================
# 2. Cargar el archivo CSV
# ================================================
# Cambia el nombre del archivo por el tuyo
df = pd.read_csv("csv_unido.csv")

# Mostrar primeras filas
print(df.head())
#display(df.head())


# ================================================
# 3. Separar atributos (X) y clases (y)
# ================================================
X = df.drop("action", axis=1)
y = df["action"]

# ================================================
# 4. Reducir dimensionalidad a 2D con PCA
# ================================================
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# ================================================
# 5. Graficar distribución de clases en 2D
# ================================================
plt.figure(figsize=(10,7))
sns.scatterplot(
    x=X_2d[:,0],
    y=X_2d[:,1],
    hue=y.astype(str),        # colores según clase
    palette="tab10",
    s=60,
    alpha=0.8
)

plt.title("Distribución de acciones (action) en 2D mediante PCA", fontsize=14)
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title="action")
plt.grid(True, alpha=0.3)

plt.show()
