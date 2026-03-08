main_monitor.py
# # Sistema de Respuesta ante Incidentes (IR) con Selección Dinámica de Modelos
# **Autor:** Gustavo Alfonso Maldonado Vallejo  
# **Objetivo:** Desarrollar un motor de decisión que optimice la detección de amenazas en red, eligiendo entre precisión máxima (CatBoost) o velocidad de respuesta (LightGBM) según el contexto del servidor.

# %%
#Importacion de librerias

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import glob
import os

print("Librerías listas para la misión.")

# %% [markdown]
# ## 1. Carga y Limpieza de Datos (Dataset CIC-IDS2017)
# En esta sección cargamos los logs de tráfico de red y realizamos una limpieza de valores infinitos, comunes en capturas de ataques de denegación de servicio (DoS).

# %%
import pandas as pd
import numpy as np
import glob
import os

# 1. Intentamos encontrar la carpeta 'data' de forma robusta
path = '../data'

# Verificamos si la carpeta existe antes de buscar los CSV
if not os.path.exists(path):
    print(f"¡Error! No encuentro la carpeta '{path}' en {os.getcwd()}")
else:
    # Buscamos archivos .csv
    all_files = glob.glob(os.path.join(path, "*.csv"))
    print(f"Archivos encontrados: {len(all_files)}")

    if len(all_files) == 0:
        print("La carpeta existe pero está vacía o los archivos no terminan en .csv")
    else:
        df_list = []
        for filename in all_files:
            # Cargamos una muestra de cada uno
            temp_df = pd.read_csv(filename, nrows=50000, low_memory=False)
            temp_df.columns = temp_df.columns.str.strip()
            df_list.append(temp_df)
            print(f"Cargado exitosamente: {os.path.basename(filename)}")

        # Ahora sí unimos
        df_security = pd.concat(df_list, axis=0, ignore_index=True)
        
        # Limpieza de nulos e infinitos
        df_security = df_security.replace([np.inf, -np.inf], np.nan).dropna()
        print(f"\n--- ÉXITO ---")
        print(f"Total de registros consolidados: {df_security.shape[0]}")

# %% [markdown]
# ## 2. Análisis Exploratorio: Distribución de Ataques

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Aseguramos la creación de la etiqueta binaria (0: Benigno, 1: Ataque)
# Nota: Usamos .str.contains porque a veces el dataset trae espacios invisibles
df_security['Is_Attack'] = df_security['Label'].apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1)

# 2. Ver qué tipos de ataques tenemos y cuántos
ataques_counts = df_security['Label'].value_counts()

# 3. Graficar (Para que se vea profesional en tu portafolio)
plt.figure(figsize=(12, 6))
sns.barplot(x=ataques_counts.values, y=ataques_counts.index, hue=ataques_counts.index, palette='viridis', legend=False)
plt.title('Distribución de Tipos de Tráfico en la Red (Dataset Consolidado)')
plt.xlabel('Número de Registros')
plt.ylabel('Tipo de Alerta')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# 4. Mostrar el porcentaje ahora que la columna EXISTE
print(f"Porcentaje de tráfico BENIGNO: {df_security['Is_Attack'].value_counts(normalize=True)[0]*100:.2f}%")
print(f"Porcentaje de tráfico de ATAQUE: {df_security['Is_Attack'].value_counts(normalize=True)[1]*100:.2f}%")

# %%
# Listamos las columnas para estar seguros de qué nombres tienen
# (Esto te ayudará a ver si hay espacios o errores de escritura)
print("Columnas disponibles:", df_security.columns.tolist())

# Seleccionamos características numéricas estándar para ciberseguridad
# Si alguna falla, revisa el print de arriba y ajusta el nombre
features_nombres = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 
    'Total Backward Packets', 'Flow IAT Mean'
]

X = df_security[features_nombres]
y = df_security['Is_Attack']

# División 80/20 con estratificación
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n--- Preparación de Datos Lista ---")
print(f"Registros de entrenamiento: {X_train.shape[0]}")
print(f"Registros de evaluación: {X_test.shape[0]}")

# %% [markdown]
# ## 3. Entrenamiento del Modelo de Baja Latencia (LightGBM)

# %%
# 1. Instanciamos el modelo
# Usamos parámetros estándar que dan un buen equilibrio entre velocidad y acierto
lgbm_model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=10,
    random_state=42,
    verbose=-1 # Para que no llene la pantalla de logs innecesarios
)

# 2. Entrenamos y medimos el tiempo (Crucial para tu comparativa de eficiencia)
print("--- Iniciando entrenamiento de LightGBM ---")
start_time = time.time()
lgbm_model.fit(X_train, y_train)
end_time = time.time()

training_time = end_time - start_time
print(f"¡Entrenamiento completado en {training_time:.4f} segundos!")

# 3. Realizamos predicciones en el set de prueba
y_pred = lgbm_model.predict(X_test)

# 4. Evaluamos el desempeño
print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred))
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Creamos la matriz
cm = confusion_matrix(y_test, y_pred)

# Graficamos de forma profesional
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benigno', 'Ataque'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Matriz de Confusión: Detección de Intrusos (LightGBM)')
plt.show()

# %% [markdown]
# ## 4. Entrenamiento del Modelo de Alta Precisión (CatBoost)

# %%
# 1. Instanciamos CatBoost
# Este modelo es famoso por manejar muy bien los datos sin tanto preprocesamiento
cat_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=10,
    random_seed=42,
    verbose=False # Para mantener el notebook limpio
)

# 2. Medimos tiempo y entrenamos
print("--- Iniciando entrenamiento de CatBoost ---")
start_time = time.time()
cat_model.fit(X_train, y_train)
end_time = time.time()

print(f"Entrenamiento completado en {end_time - start_time:.4f} segundos")

# 3. Evaluación
y_pred_cat = cat_model.predict(X_test)
print("\n--- Reporte de Clasificación (CatBoost) ---")
print(classification_report(y_test, y_pred_cat))

# %%
# Comparamos Falsos Negativos (lo más peligroso en seguridad)
cm_lgbm = confusion_matrix(y_test, y_pred)
cm_cat = confusion_matrix(y_test, y_pred_cat)

fn_lgbm = cm_lgbm[1, 0]
fn_cat = cm_cat[1, 0]

print(f"Ataques NO detectados por LightGBM: {fn_lgbm}")
print(f"Ataques NO detectados por CatBoost: {fn_cat}")

if fn_cat < fn_lgbm:
    print("\n[INFO] CatBoost demuestra ser más preciso para detectar amenazas críticas.")
else:
    print("\n[INFO] LightGBM mantiene un equilibrio óptimo entre velocidad y precisión.")

# %%
def monitor_de_respuesta(ataque_detectado, criticidad_activo, carga_cpu):
    """
    Simulación del 'Cerebro' de tu Monitor de Alertas.
    """
    print(f"\n[!] ALERTA: {ataque_detectado}")
    
    # Lógica basada en tu plan de especialización híbrida
    if criticidad_activo >= 8 and carga_cpu < 0.6:
        modelo_elegido = "CatBoost"
        estrategia = "Análisis Exhaustivo"
        # Aquí usarías cat_model.predict()
    else:
        modelo_elegido = "LightGBM"
        estrategia = "Respuesta Ultra-Rápida"
        # Aquí usarías lgbm_model.predict()
        
    print(f"[*] Activo Nivel: {criticidad_activo} | Carga Sistema: {carga_cpu*100}%")
    print(f"[->] Decisión: Usar {modelo_elegido} para {estrategia}")

# Simulación 1: Ataque a Base de Datos (Crítico) con servidor despejado
monitor_de_respuesta("Posible SQL Injection", criticidad_activo=9, carga_cpu=0.3)

# Simulación 2: Ataque masivo (DDoS) con servidor saturado
monitor_de_respuesta("DDoS detectado", criticidad_activo=5, carga_cpu=0.8)

# %%
import joblib

# Creamos una carpeta para guardar los modelos
os.makedirs('models', exist_ok=True)

# Guardamos los modelos (el 'cerebro' que ya aprendió)
joblib.dump(lgbm_model, 'models/lgbm_security_model.pkl')
joblib.dump(cat_model, 'models/catboost_security_model.pkl')

print("--- [!] Modelos exportados exitosamente en la carpeta /models ---")
print("Ahora tu Monitor de Alertas puede cargarlos en milisegundos sin reentrenar.")

# %% [markdown]
# 5. Conclusiones Estratégicas y Resultados del Engine
# ¿Cómo llegamos a estos resultados?
# Para construir este Sistema de Respuesta Dinámica, consolidamos un dataset masivo de 399,687 registros provenientes de 8 archivos distintos del benchmark CIC-IDS2017. Este proceso incluyó:
# 
# Ingeniería de Datos: Limpieza de valores infinitos y nulos, y normalización de nombres de columnas para asegurar la integridad del entrenamiento.
# 
# Entrenamiento Comparativo: Sometimos los datos a dos arquitecturas de Gradient Boosting: LightGBM (optimizado para velocidad) y CatBoost (optimizado para precisión estructural).
# 
# Evaluación de Riesgo: Más allá del Accuracy del 99%, nos enfocamos en los Falsos Negativos, que representan la amenaza real: ataques que el sistema no detectó.
# 
# Hallazgos Clave
# Superioridad Inesperada de LightGBM: En este escenario específico, LightGBM demostró ser más robusto, dejando pasar solo 395 ataques frente a los 458 de CatBoost.
# 
# Eficiencia de Recursos: LightGBM completó el entrenamiento y la inferencia en una fracción del tiempo, lo que lo posiciona como la mejor opción para activos de carga media o alta.
# 
# Resiliencia Híbrida: A pesar de los números, mantenemos a CatBoost en el motor para activos de Nivel de Criticidad >= 8. Esto se debe a que la diversidad de modelos (Ensemble Thinking) previene que un atacante experto encuentre un "punto ciego" único en nuestro firewall algorítmico.
# 
# Impacto en el Negocio (Ciberseguridad Realista)
# Este enfoque no solo detecta intrusos, sino que optimiza los costos operativos. Al delegar el tráfico masivo a un modelo ligero y reservar el análisis profundo para activos críticos, garantizamos que la infraestructura (Servidores/Home Lab) mantenga su disponibilidad sin sacrificar la seguridad.

# %%
# Fix upload 

# %%



