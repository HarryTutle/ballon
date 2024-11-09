# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 11:52:44 2024

@author: thoma
"""

""" librairies """

import numpy as np # permet le calcul tensoriel rapide.
import pandas as pd # traitement de tableaux, données, stats.
import matplotlib.pyplot as plt # graphiques.
import datetime as dt # utiliser une variable temporelle comme index.

from tensorflow.keras import Model # deep learning.
from tensorflow.keras import optimizers, layers, Sequential
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks
from tensorflow.keras import wrappers
from tensorflow.data import Dataset, TextLineDataset
from tensorflow.io import decode_csv
from tensorflow import stack, constant, float32, convert_to_tensor, float64, slice, int64
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow import reshape



from sklearn.model_selection import train_test_split # métriques, normalisation...
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, accuracy_score
from sklearn.model_selection import RandomizedSearchCV

import glob

dir_model=models.load_model('D:/station meteo 2024/lium_data/modeles/directionSE_total_model_lstm_10h_1h_6min.h5')
for_model=models.load_model('D:/station meteo 2024/lium_data/modeles/2_forceSE_total_model_lstm_10h_1h_6min.h5')

files=glob.glob('D:/station meteo 2024/lium_data/valid/2018SE*')
data=[]
for file in files:
    df=pd.read_csv(file)
    data.append(df)
data=pd.concat(data)
print(data.dtypes)
'''
def preprocess(line):
    
    
    
    defs=[0.]*1001+[constant([], dtype=float32)]+[constant([], dtype=float32)]
    fields=decode_csv(line, record_defaults=defs)
    
    x=stack(fields[1:-2])
    y=stack(fields[-1])
    
    

    x=reshape(x, [100, 10])
    x=slice(x, [0, 1], [100, 9])
        
    

cibles=data.iloc[:,-2:]
data=data.iloc[:,1:-2]
length=data.shape[0]

data=constant(data.values, dtype=float64)
data=reshape(data, [length, 100, 10])
data=slice(data, [0, 0, 1], [length, 100, 9])

cibles_dir=constant(cibles.iloc[:, 0].values, dtype=int64)
cibles_for=constant(cibles.iloc[:, 1].values, dtype=float64)

resultat_dir=dir_model.predict(data).argmax(axis=1)
resultat_for=for_model.predict(data)

print(data.dtype)
'''
data=data.drop(['Unnamed: 0'], axis=1)
dataset=data.iloc[:, :-2]

cibles_dir=np.array(data.iloc[:,-2])
cibles_for=np.array(data.iloc[:,-1])
dataset=np.array(dataset).reshape(dataset.shape[0], 100, 10)
dataset=dataset[:,:,1:]

resultat_dir=dir_model.predict(dataset).argmax(axis=1)
resultat_for=for_model.predict(dataset)

print(accuracy_score(cibles_dir, resultat_dir))
print(mean_absolute_error(cibles_for, resultat_for))

marge_erreur_dir=abs(cibles_dir-resultat_dir)
marge_erreur_for=abs(cibles_for-resultat_for[:,0])

debug=dataset[:,0,1].shape
erreur=pd.DataFrame({'latitude':list(dataset[:,0,0]),
                             'longitude':list(dataset[:,0,1]),
                             'altitude': list(dataset[:,0,2]),
                             'erreur_dir': list(marge_erreur_dir),
                             'erreur_for': list(marge_erreur_for)})


erreur['latitude']=erreur['latitude'].apply(lambda x:x*253.68+4595.35)
erreur['longitude']=erreur['longitude'].apply(lambda x: x*385.51+237.56)
erreur['altitude']=erreur['altitude'].apply(lambda x: x*179.13+142.38)

resultat=erreur.groupby('latitude').sum()


plt.figure(figsize=(20, 20))
plt.title('marge erreur selon region, direction')
plt.scatter(y=dataset[:,0,0]*253.68+4595.35, x=dataset[:,0,1]*385.51+237.56, c=marge_erreur_dir, label='erreur', s=500)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.legend()
plt.show()

plt.figure(figsize=(20, 20))
plt.title('marge erreur selon region, force')
plt.scatter(y=dataset[:,0,0]*253.68+4595.35, x=dataset[:,0,1]*385.51+237.56, c=marge_erreur_for, label='erreur', s=500)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.legend()
plt.show()