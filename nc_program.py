#Proyecto de desarrollo de un MVP de Prevencion de Fraudes ,a traves,de un 
#Algoritmo de Random Forest,utilizando Optuna
#Ruta al archivo del dataset en el Readme

import pandas as pd


df_fraude= pd.read_csv('ruta al archivo')
df_fraude_sin_nulos = df_fraude.dropna()
print(df_fraude_sin_nulos.info())


# Cambiamos tipos de datos usando .loc para evitar SettingWithCopyWarning


df_fraude_sin_nulos.loc[:, 'type'] = df_fraude_sin_nulos['type'].astype('category')
df_fraude_sin_nulos.loc[:, 'nameOrig'] = df_fraude_sin_nulos['nameOrig'].astype('category')
df_fraude_sin_nulos.loc[:, 'nameDest'] = df_fraude_sin_nulos['nameDest'].astype('category')
df_fraude_sin_nulos.loc[:, 'isFraud'] = df_fraude_sin_nulos['isFraud'].astype(bool)

# Eliminamos la columna 'isFlaggedFraud'

df_fraude_sin_nulos = df_fraude_sin_nulos.drop(columns=['isFlaggedFraud'])
df_fraude_sin_nulos = df_fraude_sin_nulos.drop(columns=['step'])

#########################################################################################################
"""

"""

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import optuna
from sklearn import metrics

"""
Un Bosque Aleatorio (Random Forest) es un algoritmo de aprendizaje automático que combina 
múltiples árboles de decisión para mejorar la precisión y la robustez de las predicciones.
Se basa en la idea de que un conjunto de árboles diversificados puede ofrecer mejores resultados
que un solo árbol.

Cuando enviamos datos a cualquier modelo de aprendizaje automático (ML), debemos hacerlo en el
formato adecuado, ya que los algoritmos solo entienden números.

En este enfoque, a cada etiqueta se le asigna un número entero único según el orden alfabético.
implementamos esto usando la biblioteca Scikit-learn.
"""
le = LabelEncoder()
df_fraude_sin_nulos['type'] = le.fit_transform(df_fraude_sin_nulos['type'])
df_fraude_sin_nulos['nameOrig'] = le.fit_transform(df_fraude_sin_nulos['nameOrig'])
df_fraude_sin_nulos['nameDest'] = le.fit_transform(df_fraude_sin_nulos['nameDest'])

#print(df_fraude_sin_nulos.info())

#Train - Split

"""
Para optimizar el modelo de clasificacion,Random forest vamos a usar un software llamado Optuna.

Optuna es una biblioteca de Python para la optimización de hiperparámetros.
Permite automatizar la búsqueda de la mejor configuración de un modelo de
aprendizaje automático, evaluando diferentes valores de los hiperparámetros
y seleccionando la combinación que optimiza un criterio específico.

Entre las ventajas de usar optuna podemos mencionar brevemente:
  •	Mejora del rendimiento del modelo.
  •	Ahorro de tiempo.
  •	Eficiencia.
  •	Escalabilidad.
  •	Facilidad de uso.
  •	Independiente de la plataforma.
   
"""

class modelOptimization:
    
    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test
        
    def objective(self,trial):  
    
        criterio          = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        n_estimators      = trial.suggest_int('n_estimators', 2, 50)
        max_depth         = trial.suggest_int('max_depth', 5, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 8)
        min_samples_leaf  = trial.suggest_int('min_samples_leaf', 1, 8)
        max_features      = trial.suggest_int('max_features',1,20)
        
        model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf,random_state=42,
                                       criterion=criterio,class_weight = 'balanced',
                                       max_features = max_features, n_jobs=-1)
     
        model.fit(self.X_train, self.y_train)
        
        return model.score(self.X_test, self.y_test)

if __name__ == '__main__':
    
    dataf = df_fraude_sin_nulos[:] 

    X = dataf.drop("isFraud", axis = 1)
    y = dataf["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    model = modelOptimization(X_train, X_test, y_train, y_test)

    study = optuna.create_study(direction='maximize')

    study.optimize(model.objective, n_jobs=-1, n_trials=10)

    print(f"Best Parameters : {study.best_params}")
    print(f"Best Score      : {study.best_value}")

"""
Las Salidas generadas por el modelo de acuerdo a Optuna ,serian las siguientes:
    
A new study created in memory with name: no-name-ae87226b-eb5b-47e4-a861-850076f15f3c
•	Trial 3 finished with value: 0.9970368600775571 and parameters: 
{'criterion': 'gini', 'n_estimators': 24, 'max_depth': 19, 'min_samples_split': 5,
 'min_samples_leaf': 6, 'max_features': 2}. Best is trial 3 with value: 0.9970368600775571.

•	Trial 1 finished with value: 0.9974743108970833 and parameters: 
{'criterion': 'gini', 'n_estimators': 14, 'max_depth': 14, 'min_samples_split': 8,
 'min_samples_leaf': 2, 'max_features': 9}. Best is trial 1 with value: 0.9974743108970833.

•	Trial 2 finished with value: 0.9974198260045914 and parameters:
 {'criterion': 'gini', 'n_estimators': 45, 'max_depth': 17, 'min_samples_split': 3, 
  'min_samples_leaf': 5, 'max_features': 3}. Best is trial 1 with value: 0.9974743108970833.

•	Trial 0 finished with value: 0.9995709314716265 and parameters: 
{'criterion': 'entropy', 'n_estimators': 23, 'max_depth': 18, 'min_samples_split': 7, 
 'min_samples_leaf': 8, 'max_features': 16}. Best is trial 0 with value: 0.9995709314716265.

•	Trial 6 finished with value: 0.9992204469228085 and parameters:
 {'criterion': 'gini', 'n_estimators': 6, 'max_depth': 18, 'min_samples_split': 8, 
  'min_samples_leaf': 4, 'max_features': 17}. Best is trial 0 with value: 0.9995709314716265.

•	Trial 8 finished with value: 0.9627061388757042 and parameters: 
{'criterion': 'entropy', 'n_estimators': 22, 'max_depth': 5, 'min_samples_split': 5, 
 'min_samples_leaf': 7, 'max_features': 1}. Best is trial 0 with value: 0.9995709314716265.

•	Trial 4 finished with value: 0.9974889799066003 and parameters:
 {'criterion': 'gini', 'n_estimators': 42, 'max_depth': 15, 'min_samples_split': 5, 
  'min_samples_leaf': 4, 'max_features': 5}. Best is trial 0 with value: 0.9995709314716265.

•	Trial 5 finished with value: 0.9919959597356645 and parameters: 
{'criterion': 'entropy', 'n_estimators': 43, 'max_depth': 12, 'min_samples_split': 6, 
 'min_samples_leaf': 3, 'max_features': 3}. Best is trial 0 with value: 0.9995709314716265.

•	Trial 9 finished with value: 0.9974161587522121 and parameters:
 {'criterion': 'gini', 'n_estimators': 37, 'max_depth': 16, 'min_samples_split': 7,
  'min_samples_leaf': 1, 'max_features': 4}. Best is trial 0 with value: 0.9995709314716265.

•	Trial 7 finished with value: 0.993802867372246 and parameters: 
{'criterion': 'entropy', 'n_estimators': 41, 'max_depth': 9, 'min_samples_split': 2,
 'min_samples_leaf': 2, 'max_features': 10}. Best is trial 0 with value: 0.9995709314716265.

•	Best Parameters : 
{'criterion': 'entropy', 'n_estimators': 23, 'max_depth': 18, 'min_samples_split': 7,
 'min_samples_leaf': 8, 'max_features': 16}

•	Best Score      : 0.9995709314716265

"""
#Visualizacion de la optimizacion de hiperparametros

import matplotlib.pyplot as plt

optuna.visualization.matplotlib.plot_optimization_history(study)
plt.tight_layout()


##########################################################################################
#Evaluar el Modelo:
"""
1
Resumen de Métricas para Evaluar un Modelo de Random Forest
1. Precisión (Accuracy): Mide la proporción de predicciones correctas realizadas por el modelo.
                        Es una métrica simple, pero puede ser engañosa en conjuntos de datos desbalanceados.

2. Precisión (Precision): Mide la proporción de predicciones positivas que son realmente correctas. 
                        Es útil para evaluar la capacidad del modelo para identificar correctamente 
                        casos positivos.

3. Recall (Sensibilidad): Mide la proporción de casos positivos que son correctamente identificados
                          por el modelo. Es útil para evaluar la capacidad
                          del modelo para no omitir casos positivos.

4. F1 Score: Es una métrica que combina la precisión y el recall en una sola medida.
             Es útil para obtener una visión general del rendimiento del modelo.

5. Matriz de Confusión: Es una tabla que muestra la relación entre las predicciones del 
                       modelo y las etiquetas reales. Es útil para visualizar el rendimiento
                       del modelo y para identificar posibles errores.

6. Curva ROC y AUC (Área Bajo la Curva ROC): La curva ROC muestra la tasa de verdaderos positivos
                                             frente a la tasa de falsos positivos a diferentes umbrales
                                             de clasificación. El AUC es una medida del rendimiento 
                                             general del modelo para clasificar casos positivos y negativos.

"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Cargar el modelo y los datos con los valores optimizados por Optuna

model = RandomForestClassifier(n_estimators=23,max_depth=18,
                               min_samples_split=7,
                               min_samples_leaf=8,random_state=42,
                               criterion='entropy',class_weight = 'balanced',
                               max_features = 16, n_jobs=-1)
model.fit(X_train, y_train)
# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar las métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# Imprimir los resultados
print("Accuracy :", round(accuracy,ndigits=4))
print("Precisión:", round(precision,ndigits=4))
print("Recall   :", round(recall,ndigits=4))
print("F1 Score :",round(f1,ndigits=4))

"""

(Accuracy): Mide la proporción de predicciones correctas realizadas por el modelo.
Accuracy : 0.9996

(Precision): Mide la proporción de predicciones positivas que son realmente correctas. 
                        Es útil para evaluar la capacidad del modelo para identificar correctamente 
                        casos positivos.
Precisión: 0.7674

Recall (Sensibilidad): Mide la proporción de casos positivos que son correctamente identificados
                          por el modelo. Es útil para evaluar la capacidad
                          del modelo para no omitir casos positivos.
Recall   : 0.9524

F1 Score: Es una métrica que combina la precisión y el recall en una sola medida.
             Es útil para obtener una visión general del rendimiento del modelo.
F1 Score : 0.8499

"""

#Visualizar las Principales Metricas del Modelo 

from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.plot(recall, precision, label="F1 Score = " + str(f1))
plt.xlabel("Recall")
plt.ylabel("Precisión")
plt.title("Curva Precisión-Recall")
plt.legend()
plt.show()


# Calcular la precisión y la sensibilidad a diferentes umbrales

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

# Graficar la curva de precisión vs sensibilidad
plt.plot(recall, precision)
plt.xlabel("Sensibilidad")
plt.ylabel("Precisión")
plt.title("Curva de Precisión vs Sensibilidad")
plt.show()


conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:", conf_matrix)

# Crear la visualización de la matriz de confusión
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix,                                             
                                            display_labels = [False, True])

# Graficar la matriz de confusión
cm_display.plot()
plt.show()


###################################################################################

# Obtener las predicciones del modelo
y_pred = model.predict(X_test)

# Calcular la matriz de confusión
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

# Calcular PPV y FPR
tn, fp, fn, tp = confusion_matrix.ravel()
ppv = tp / (tp + fp)
fpr = fp / (fp + tn)

# Imprimir los resultados
print("PPV:",round(ppv,ndigits=2))
print("FPR:", fpr)


# Obtener las métricas ROC
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
auc_roc = metrics.auc(fpr, tpr)
print("Valor AUC-ROC",auc_roc)

# Graficar la curva ROC
plt.plot(fpr, tpr, label="AUC-ROC = " + str(auc_roc))
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC")
plt.legend()
plt.show()


"""
Recall: Un valor de 0.9524 indica que el modelo identifica correctamente el 95.24% de las 
transacciones fraudulentas. 

El modelo muestra un buen equilibrio entre la capacidad de detectar fraudes y la generación de falsos positivos.

El valor de AUC-ROC (0.976) indica un buen rendimiento general del modelo para la detección
 de fraude en un conjunto de datos desbalanceado.AUC-ROC (Área Bajo la Curva ROC) mide la 
capacidad del modelo para diferenciar entre las clases positiva (fraude) y negativa 
(transacción legítima) a diferentes umbrales de clasificación. Un valor de 1 indica una
 perfecta diferenciación, mientras que un valor de 0.5 indica que el modelo no es mejor 
 que adivinar al azar.En este caso, un AUC-ROC de 0.976 significa que el modelo es bastante
 bueno para discriminar entre transacciones fraudulentas y legítimas en todo el rango de 
 posibles umbrales de clasificación.
 
 F1 Score de 0.8499 indica que el modelo es capaz de realizar la tarea de clasificación
 con un alto grado de precisión y exhaustividad.
 
"""
##############################################################################################

#Buscamos Optimizar el Threshold de nuestro modelo ,mediante optuna
# Entrena el modelo Random Forest
clf =RandomForestClassifier(n_estimators=23,max_depth=18,
                               min_samples_split=7,
                               min_samples_leaf=8,random_state=42,
                               criterion='entropy',class_weight = 'balanced',
                               max_features = 16, n_jobs=-1).fit(X_train, y_train)

# Predice las probabilidades
y_proba = clf.predict_proba(X_test)[:, 1]  # Probabilidad de clase positiva


# Importamos las librerías necesarias

from optuna import create_study
from sklearn.metrics import accuracy_score

# Función objetivo para Optuna
def objective(trial):
  """Función objetivo para Optuna."""
  
  threshold = trial.suggest_float("threshold", 0.0, 1.0)
  y_pred = (y_proba>= threshold).astype(int)
  accuracy = accuracy_score(y_test, y_pred)
  return accuracy

# Definimos el estudio de Optuna
study = create_study(direction="maximize")

# Búsqueda del mejor umbral
study.optimize(objective, n_trials=100)

# Obtenemos el mejor valor de umbral
best_threshold = study.best_trial.params["threshold"]

# Imprimimos el mejor valor de umbral y la precisión
print(f"Mejor umbral: {best_threshold}")
print(f"Precision   : {study.best_value}")

###########################################################################################################
###########################################################################################################

# Predice las probabilidades
y_proba = model.predict_proba(X_test)[:, 1]  # Probabilidad de clase positiva

# Definimos el umbral
threshold = 0.7822

# Obtenemos las predicciones del modelo
y_pred = (y_proba >= threshold).astype(int)

conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:", conf_matrix)

# Crear la visualización de la matriz de confusión
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix,                                             
                                            display_labels = [False, True])

# Graficar la matriz de confusión
cm_display.plot()
plt.show()


# Calculamos la precisión, la revocación y la puntuación F1
precision, recall, _ = precision_recall_curve(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Mostramos la información
print(f"Precisión    : {precision[-1]:.4f}")
print(f"Recall       : {recall[-1]:.4f}")
print(f"Puntuación F1: {f1:.4f}")

# Creamos la curva ROC
plt.plot(recall, precision, label="Curva ROC")
plt.xlabel("Recall")
plt.ylabel("Precisión")
plt.legend()
plt.show()



# Obtener las métricas ROC
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
auc_roc = metrics.auc(fpr, tpr)
print("Valor AUC-ROC",auc_roc)

# Graficar la curva ROC
plt.plot(fpr, tpr, label="AUC-ROC = " + str(auc_roc))
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC")
plt.legend()
plt.show()


from sklearn.metrics import precision_recall_fscore_support

def calcular_metricas_desbalanceadas(y_true, y_pred):
  """
  Calcula métricas específicas para el desbalanceo en un modelo de detección de fraude.

  Args:
    y_true: Etiquetas reales.
    y_pred: Predicciones del modelo.

  Returns:
    Diccionario con las métricas:
      * precision_por_clase: Precisión por clase.
      * recall_por_clase: Revocación por clase.
      * f1_score_por_clase: F1-score por clase.
      * confusion_matrix: Matriz de confusión.
  """
 
  #Metricas por clase
  # Calcular la precisión, la revocación y la F1-score por clase
  
  precision, recall, f1_score, support = precision_recall_fscore_support(y_true, y_pred)

  # Calcular la matriz de confusión
  conf_matrix = metrics.confusion_matrix(y_test, y_pred)

  # Generar un diccionario con las métricas
  metricas = {
      "precision_por_clase": precision,
      "recall_por_clase": recall,
      "f1_score_por_clase": f1_score,
      "conf_matrix"    : conf_matrix,
  }

  return metricas

metricas = calcular_metricas_desbalanceadas(y_test, y_pred)

# Imprimir las métricas
print(f"Precisión por clase: {metricas['precision_por_clase']}")
print(f"Revocación por clase: {metricas['recall_por_clase']}")
print(f"F1-score por clase: {metricas['f1_score_por_clase']}")
print(f"Matriz de confusión: {metricas['conf_matrix']}")








