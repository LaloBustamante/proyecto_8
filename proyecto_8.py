'''
Descripción del proyecto

La compañía móvil Megaline no está satisfecha al ver que muchos de sus clientes utilizan planes heredados. 
Quieren desarrollar un modelo que pueda analizar el comportamiento de los clientes y recomendar uno de los nuevos planes de Megaline: Smart 
o Ultra.

Tienes acceso a los datos de comportamiento de los suscriptores que ya se han cambiado a los planes nuevos (del proyecto del sprint de 
Análisis estadístico de datos). Para esta tarea de clasificación debes crear un modelo que escoja el plan correcto. Como ya hiciste el paso 
de procesar los datos, puedes lanzarte directo a crear el modelo.

Desarrolla un modelo con la mayor exactitud posible. En este proyecto, el umbral de exactitud es 0.75. Usa el dataset para comprobar la 
exactitud.

Instrucciones del proyecto.

1) Abre y examina el archivo de datos. Dirección al archivo:datasets/users_behavior.csv Descarga el dataset

2) Segmenta los datos fuente en un conjunto de entrenamiento, uno de validación y uno de prueba.

3) Investiga la calidad de diferentes modelos cambiando los hiperparámetros. Describe brevemente los hallazgos del estudio.

4) Comprueba la calidad del modelo usando el conjunto de prueba.

5) Tarea adicional: haz una prueba de cordura al modelo. Estos datos son más complejos que los que habías usado antes así que no será una tarea 
fácil. Más adelante lo veremos con más detalle.

Descripción de datos
Cada observación en el dataset contiene información del comportamiento mensual sobre un usuario. La información dada es la siguiente:

calls — número de llamadas,
minutes — duración total de la llamada en minutos,
messages — número de mensajes de texto,
mb_used — Tráfico de Internet utilizado en MB,
is_ultra — plan para el mes actual (Ultra - 1, Smart - 0)

Evaluación del proyecto

Hemos definido los criterios de evaluación para el proyecto. Lee esto con atención antes de pasar al ejercicio. 

Esto es lo que los revisores buscarán cuando evalúen tu proyecto:

¿Cómo leíste los datos después de descargarlos?
¿Segmentaste correctamente los datos en conjuntos de entrenamiento, validación y prueba?
¿Cómo escogiste el tamaño de los conjuntos?
¿Evaluaste correctamente la calidad del modelo?
¿Qué modelos e hiperparámentros usaste?
¿Cuáles fueron tus hallazgos?
¿Probaste los modelos correctamente?
¿Cuál es tu puntuación de exactitud?
¿Te ceñiste a la estructura del proyecto y mantuviste limpio el código?

'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos:
df = pd.read_csv('/datasets/users_behavior.csv')


# Mostrar las primeras filas para tener una idea general de los datos
print(df.head())

# Verificar información básica sobre los tipos de datos y valores faltantes
print(df.info())

# Obtener un resumen estadístico de los datos
print(df.describe())

# Visualizar los datos
# Histograma para las columnas numéricas
df[['calls', 'minutes', 'messages', 'mb_used']].hist(bins=30, figsize=(10, 8))
plt.show()

'''
El siguiente paso en el proyecto es segmentar los datos en un conjunto de entrenamiento, uno de validación y uno de prueba.

Conjunto de entrenamiento (training set): Se usará para entrenar el modelo.
Conjunto de validación (validation set): Se usará para ajustar los hiperparámetros del modelo.
Conjunto de prueba (test set): Se usará al final para evaluar la calidad del modelo de forma objetiva.
'''

#  División de los datos en conjuntos de entrenamiento, validación y prueba

# Definir las características (features) y el objetivo (target)
features = df.drop(['is_ultra'], axis=1)
target = df['is_ultra']

# Dividir el dataset en conjunto de entrenamiento (60%) y conjunto de prueba (40%)
features_train, features_temp, target_train, target_temp = train_test_split(features, target, test_size=0.4, random_state=12345)

# Dividir el conjunto temporal (40%) en conjunto de validación (50% de los 40%) y conjunto de prueba (50% de los 40%)
features_valid, features_test, target_valid, target_test = train_test_split(features_temp, target_temp, test_size=0.5, random_state=12345)

# Verificar tamaños de los conjuntos
print("Tamaño del conjunto de entrenamiento:", features_train.shape)
print("Tamaño del conjunto de validación:", features_valid.shape)
print("Tamaño del conjunto de prueba:", features_test.shape)

'''
Ahora con los datos segmentados, es posible empezar a entrenar diferentes modelos y evaluar su rendimiento utilizando el conjunto de 
validación.
'''

'''
El siguiente paso en el proyecto es investigar la calidad de diferentes modelos cambiando los hiperparámetros, y describir los hallazgos. 
Ahora el objetivo es entrenar varios modelos de clasificación y encontrar el que ofrezca la mejor exactitud (accuracy) en el conjunto de 
validación.

Entrenar modelos de clasificación:

Árboles de decisión (Decision Trees)
Bosques aleatorios (Random Forests)
Regresión logística (Logistic Regression)
'''

'''
Árbol de Decisión (DecisionTreeClassifier):
Aqui se itera sobre diferentes profundidades del árbol (max_depth) para encontrar el valor que maximice la precisión en el conjunto de 
validación.
'''
# 1. Árbol de Decisión
for depth in range(1, 6):
    model_tree = DecisionTreeClassifier(max_depth=depth, random_state=12345)
    model_tree.fit(features_train, target_train)
    predictions_valid = model_tree.predict(features_valid)
    accuracy = accuracy_score(target_valid, predictions_valid)
    print(f"Decision Tree (max_depth={depth}): accuracy = {accuracy:.4f}")


'''
Bosque Aleatorio (RandomForestClassifier):
Se cambia el número de árboles (n_estimators) para ver cómo afecta la precisión. El modelo de bosque aleatorio suele ser más robusto que el 
árbol de decisión simple, pero también es más complejo.
'''
# 2. Bosque Aleatorio
for estimators in [10, 50, 100]:
    model_forest = RandomForestClassifier(n_estimators=estimators, random_state=12345)
    model_forest.fit(features_train, target_train)
    predictions_valid = model_forest.predict(features_valid)
    accuracy = accuracy_score(target_valid, predictions_valid)
    print(f"Random Forest (n_estimators={estimators}): accuracy = {accuracy:.4f}")


'''
Regresión Logística (LogisticRegression):
Este modelo de clasificación lineal tiende a ser más simple pero efectivo. Se ajusta el número máximo de iteraciones con max_iter.
'''
# 3. Regresión Logística
model_logistic = LogisticRegression(random_state=12345, max_iter=1000)
model_logistic.fit(features_train, target_train)
predictions_valid = model_logistic.predict(features_valid)
accuracy = accuracy_score(target_valid, predictions_valid)
print(f"Logistic Regression: accuracy = {accuracy:.4f}")




