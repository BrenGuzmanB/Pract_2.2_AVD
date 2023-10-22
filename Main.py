"""
Created on Fri Oct 20 18:12:15 2023

@author: Bren Guzmán, Brenda García, María José Merino
"""
#%% LIBRERÍAS
import pandas as pd
import numpy as np
from PCA import Principal_component
import matplotlib.pyplot as plt
import seaborn as sns

#%% CARGAR ARCHIVO
columns_names = ["class", "Alcohol", "Malicacid", "Ash", "Alcalinity_of_ash", "Magnesium", "Total_phenols", "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "0D280_0D315_of_diluted_wines", "Proline"]
df = pd.read_csv("Wine.csv", names= columns_names )

'''
Los atributos son:
1) Alcohol
2) ácido málico
3) Ceniza
4) Alcalinidad de la ceniza
5) magnesio
6) fenoles totales
7) Flavonoides
8) Fenoles no flavonoides
9) Proantocianinas
10) intensidad del color
11)Tono
12)OD280/OD315 de vinos diluidos
13)prolina
'''

#%% SEPARAR X, Y
# 1. Tomar el conjunto de d+1 dimensiones y descartar la información de las etiquetas/clases
X = df.drop(columns=["class"])
Y = df["class"]


# 2. Calcular la media para cada dimensión para normalizar

# Calcular la media por columna
media_por_columna = np.mean(X, axis=0)

# Restar la media a cada columna para normalizar
X_normalizado = X - media_por_columna

# Calcular la desviación estándar por columna
desviacion_estandar = np.std(X_normalizado, axis=0)

# Dividir por la desviación estándar para escalar a la unidad
X_normalizado = X_normalizado / desviacion_estandar

# 3 a 6 se encuentran dentro de la clase Principal_component en el archivo PCA
