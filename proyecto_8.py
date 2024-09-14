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