"Tipo de cambio Peso / Dolar, practica"

import numpy as np
import pandas as pd
import tensorflow as tf 
from tensorflow import  keras #algebra lineal
import seaborn as sns
from pylab import rcParams
import matplotlib.pylab as plt
from matplotlib import rc
import seaborn as sns

df = pd.read_csv("C:\\Users\\creep\\.spyder-py3\\stocks\\DEXMXUS.csv") 
df.head(50) #carga los primeros 50 elementos
df.tail(10) #carga los ultimos 100 elementos

#CONFIGURACION PARA ESTILIZAR EL GRAFICO CON PARAMETROS 
"ERROR EN LA GRAFICA"
#sns.set(style="whitegrid", palette= "muted", front_scale=1.05)
#rcParams["figure.figsize"] = 16,5

# CONFIGURACIÓN PARA ESTILIZAR EL GRÁFICO CON PARÁMETROS
sns.set(style="whitegrid", palette="muted")
rcParams["figure.figsize"] = 16,5


#valor aleatorio definido
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

#definir los datos como una serie de tiempo
df["DEXMXUS"] = pd.to_numeric(df["DEXMXUS"], errors='coerce')  # Asegúrate de que la columna de valores sea numérica
df["DATE"] = pd.to_datetime(df["DATE"])
df.index=df["DATE"]

print(df.dtypes)
df.plot(subplots = True)

df.index =df["DATE"]

features=df["DEXMXUS"]
features=features.to_frame()
features.head(15)

features = df["DATE"]

#se define el conjunto de entrenamiento
train_size=int(len(features)*0.8)
test_size=len(features)-train_size
train, test= features[0:train_size], features[train_size:len(features)]
print(len(train),len(test))