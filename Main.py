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

#%% GRÁFICA DE SCREE

PCA = Principal_component()
PCA.fit(X_normalizado)

explained_variance_ratio = PCA.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Acumulada Explicada')
plt.title('Scree Plot')
plt.grid()
plt.show()

# Determinar cuántos componentes son necesarios para mantener el 95% de la varianza
n_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Número de componentes necesarios para el 95% de la varianza: {n_components}")

#%% PCA
# Reducir la dimensionalidad a n_components
PCA = Principal_component(n_components=n_components)
PCA.fit(X_normalizado)
transformed_data = PCA.transform(X_normalizado)

# Matriz de covarianza
print("Matriz de Covarianza:")
print(PCA.covariance_matrix)

# Valores y vectores propios
print("Componentes principales:")
print(PCA.components)

print("Valores propios:")
print(PCA.eigenvalues)

# Datos tranformados con la matriz
print("Datos transformados:")
print(transformed_data)

#%% GRÁFICAS DE COMPONENTES PRINCIPALES
#%%% 3D
componentes_principales_df = pd.DataFrame(transformed_data[:, :3], columns=['Componente 1', 'Componente 2', 'Componente 3'])

# Concatena el DataFrame de componentes principales con la Serie Y
resultado = pd.concat([Y, componentes_principales_df], axis=1)

# Crea una figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Lista de etiquetas de clase únicas
clases_unicas = resultado['class'].unique()

# Asigna un color distinto a cada clase
colores = ['b', 'g', 'r']

# Itera a través de cada clase y plotea los puntos con el color correspondiente
for i, clase in enumerate(clases_unicas):
    datos_clase = resultado[resultado['class'] == clase]
    ax.scatter(datos_clase['Componente 1'], datos_clase['Componente 2'], datos_clase['Componente 3'], c=colores[i], label=f'Clase {clase}')

# Etiquetas de los ejes
ax.set_xlabel('Componente 1')
ax.set_ylabel('Componente 2')
ax.set_zlabel('Componente 3')

# Leyenda
ax.legend()

# Mostrar la gráfica
plt.show()

#%%% 2D

# Combinación de componentes y títulos
combinaciones = [(1, 2, 'Componente 1 vs. Componente 2'),
                (1, 3, 'Componente 1 vs. Componente 3'),
                (2, 3, 'Componente 2 vs. Componente 3')]

# Itera a través de las combinaciones y plotea las gráficas
for i, (comp_x, comp_y, titulo) in enumerate(combinaciones):
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    for j, clase in enumerate(clases_unicas):
        datos_clase = resultado[resultado['class'] == clase]
        ax.scatter(datos_clase[f'Componente {comp_x}'], datos_clase[f'Componente {comp_y}'], c=colores[j], label=f'Clase {clase}')
    ax.set_xlabel(f'Componente {comp_x}')
    ax.set_ylabel(f'Componente {comp_y}')
    ax.set_title(titulo)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
#%% MODELO DE CLASIFICACIÓN (REGRESIÓN LOGÍSTICA)

#%%% Componentes principales
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix


X_train, X_test, y_train, y_test = train_test_split(transformed_data, Y, test_size=0.3, random_state=5)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_pca = accuracy_score(y_test, y_pred)
classification_report_pca = classification_report(y_test, y_pred)
confusion_pca = confusion_matrix(y_test, y_pred)


#%%% Datos originales
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=5)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_og = accuracy_score(y_test, y_pred)
classification_report_og = classification_report(y_test, y_pred)
confusion_og = confusion_matrix(y_test, y_pred)


#%%% Evaluación

print('_' * 55)  
print('Resultados con los componentes principales:')
print(f'Precisión: {accuracy_pca}')
print(f'Informe de clasificación:\n{classification_report_pca}')

sns.heatmap(confusion_pca, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas Reales')
plt.title('Matriz de Confusión (PCA)')
plt.show()


print('_' * 55)  
print('Resultados con los datos originales:')
print(f'Precisión: {accuracy_og}')
print(f'Informe de clasificación:\n{classification_report_og}')

sns.heatmap(confusion_og, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas Reales')
plt.title('Matriz de Confusión (Datos originales)')
plt.show()
