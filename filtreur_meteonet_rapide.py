# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:31:46 2024

@author: thoma
"""

""" We use those packages. """

import datetime as dt
import numpy as np
import pandas as pd

zone=[4600, 4700, 120, 125]
chunksize=10000
chunks=[]
'''
for chunk in pd.read_csv('/home/harry/Documents/projet_station_meteo/NW_Ground_Stations/Meteo_NW_2016.csv',chunksize=chunksize):
    chunks.append(chunk)

for chunk in pd.read_csv('/home/harry/Documents/projet_station_meteo/NW_Ground_Stations/Meteo_NW_2017.csv',chunksize=chunksize):
    chunks.append(chunk)
'''
for chunk in pd.read_csv('C:/Users/thoma/Documents/meteo/Meteo_NW_2016.csv',chunksize=chunksize):
 
    chunk['lat']=chunk['lat'].apply(lambda x: int(x*100))
    chunk['lon']=chunk['lon'].apply(lambda x: int(x*100))
    chunk=chunk.loc[chunk.lat>zone[0] , ]
    chunk=chunk.loc[chunk.lat<zone[1] , ]
    #chunk=chunk.loc[chunk.lon>zone[2] , ]
    #chunk=chunk.loc[chunk.lon<zone[3] , ]
    chunks.append(chunk)


data=pd.concat(chunks, axis=0)
#data=data.iloc[:500,]

print(data.lat.dtype)
print(data.lon.dtype)
print(data['number_sta'].unique())
#data=data[data.lat>49.3 & data.lat<49.4 & data.lon>1.10 & data.lon<1.2]


""" This function is useful to turn wind direction variable into categorical."""

def cap(var):  
    
    if (var>337.5) or (var<=22.5):
        var=0
    elif (var>22.5) and (var<=67.5):
        var=45
    elif (var>67.5) and (var<=112.5):
        var=90
    elif (var>112.5) and (var<=157.5):
        var=135
    elif (var>157.5) and (var<=202.5):
        var=180
    elif (var>202.5) and (var<=247.5):
        var=225
    elif (var>247.5) and (var<=292.5):
        var=270
    elif (var>292.5) and (var<=337.5):
        var=315
        
    return var


""" This next function convert wind strength in knots and turn this variable into categorical."""

def vent(var):
    
    var=var*3600//1852
    
    if (var>=0) and (var<5):
        var=1
    
    elif (var>=5) and (var<10):
        var=2
        
    elif (var>=10) and (var<15):
        var=3
        
    elif (var>=15) and (var<20):
        var=4
        
    elif (var>=20) and (var<25):
        var=5
        
    elif (var>=25) and (var<30):
        var=6
        
    else:
        var=7
        
    return var


""" This another function takes care of rainfall and turn it into binary variable."""

def flotte(var):
    
    if var==0:
        var=0
        
    elif var!=0:
        var=1
        
    return var




        

""" This class create an object able to organise data from MeteoNet, ready for sklearn (see the MTTT notice for more accuracy). """


    
def Meteonet_manip(data=data, fréquence=6, heures_passé=24, minutes_futur=6, corbeille=['pluie'], variable_cible=['direction']):
        
        """
        
        fréquence: densité des échantillons (par 6 minutes par défaut)
        
        heures_passé: taille de la séquence temporelle en heures
        
        minutes_futur: prédiction dans le futur en minutes
        
        corbeille: liste des variables inutiles
        
        variable_cible: variables cibles sélectionnées
        
    
        
        """
        
        
        
    
       
        liste_finale_variables=[]
        liste_finale_cibles=[]
        

        indexage_heures=list(pd.date_range('2016-01-01 00:00:00', '2018-12-31 23:00:00', freq=str(fréquence)+'min')) # we write a datetime index on hours.
        time_heures=pd.DataFrame({'temps': indexage_heures}) 
        time_heures=time_heures.set_index('temps') 
        
        indexage_jours=list(pd.date_range('2016-01-01', '2018-12-31', freq='d')) # the same but now on days.
        time_jours=pd.DataFrame({'temps': indexage_jours})

        for station in data['number_sta'].unique(): # here for each station we organise data in one hour per row in the good time sense without any hole, and we turn it in 'int' to use less power without losing information.
           print(station)
           station_data=data.loc[data['number_sta']==station]
           station_data=station_data.sort_values(['date'],ascending=True)
           station_data=station_data.set_index('date')
           station_data.index=pd.to_datetime(station_data.index)
           station_data=station_data.resample(str(fréquence)+'min').mean()
           station_data=time_heures.join(station_data, how='outer')
           station_data["dd"]=station_data["dd"].map(lambda x: cap(x))
           station_data['number_sta']=station_data['number_sta'].apply(lambda x: int(x) if np.isnan(x)==False else x)
           #station_data['lat']=station_data['lat'].apply(lambda x: int(x*100) if np.isnan(x)==False else x)
           #station_data['lon']=station_data['lon'].apply(lambda x:int(x*100) if np.isnan(x)==False else x)
           station_data['height_sta']=station_data['height_sta'].apply(lambda x:int(x) if np.isnan(x)==False else x)
           station_data['dd']=station_data['dd'].apply(lambda x:int(x) if np.isnan(x)==False else x)
           station_data['ff']=station_data['ff'].apply(lambda x:int(round(x)) if np.isnan(x)==False else x)
           station_data['hu']=station_data['hu'].apply(lambda x:int(round(x)) if np.isnan(x)==False else x)
           station_data['precip']=station_data['precip'].apply(lambda x:int(round(x)) if np.isnan(x)==False else x)
           station_data['psl']=station_data['psl'].apply(lambda x:int(x/100) if np.isnan(x)==False else x)
           station_data['td']=station_data['td'].apply(lambda x:int(round(x)) if np.isnan(x)==False else x)
           station_data['t']=station_data['t'].apply(lambda x:int(round(x)) if np.isnan(x)==False else x)
           station_data.columns=['station', 'latitude', 'longitude', 'altitude', 'direction', 'force', 'pluie', 'humidité', 'point_rosée', 'température', 'pression']
           station_data['mois']=station_data.index.month
           station_data=station_data.drop(corbeille, axis=1)
           station_data=station_data.dropna(axis=1, how='all')
           station_data=station_data.dropna(axis=0, how='any')
           station_data=time_heures.join(station_data, how='outer')
           

               
           échelle_passé=dt.timedelta(hours=heures_passé)-dt.timedelta(minutes=fréquence)
           échelle_futur=dt.timedelta(minutes=minutes_futur)
           
                   
           for date in station_data.index:
                   
               début=date-échelle_passé
               fin=date+échelle_futur
               début_index=pd.date_range(start=début, end=date, freq=str(fréquence)+'min')
               début_index=pd.DataFrame(début_index).set_index(0)
               dataset=début_index.join(station_data, how='left')
                   
                   
               try: 
                       
                   cible=station_data.loc[fin, variable_cible]
                       
               except:
                       
                   cible=np.empty((1, len(variable_cible)))
                   cible=cible.fill(np.nan)
                   cible=pd.DataFrame(cible)
                   
               if (dataset.isnull().values.any()==False) and (cible.isnull().values.any()==False) and (cible.shape[0]==len(variable_cible)) and (dataset.shape[0]==(heures_passé*60)//fréquence) and (dataset.shape[1]==12-len(corbeille)):
                       
                   dataset=np.array(dataset)
                       
                   dataset=dataset.reshape(1, (heures_passé*60)//fréquence, 12-len(corbeille))
                       
                   dataset=np.flip(dataset, axis=1)
                   liste_finale_variables.append(dataset)
                   liste_finale_cibles.append(np.array(cible).reshape(1, len(variable_cible)))
                   
                       
               else:
                   
                   pass
                 
                       
        
        liste_finale_variables=np.concatenate(liste_finale_variables, axis=0)
        liste_finale_cibles=np.concatenate(liste_finale_cibles, axis=0)
        
        return liste_finale_variables, liste_finale_cibles

        
                              
              

dataset, cibles=Meteonet_manip(heures_passé=10, corbeille=['point_rosée', 'pluie'], variable_cible=['direction', 'force'], minutes_futur=6, fréquence=6)

np.save('2016NW4647_total_dataset_6min_10h_6min.npy', dataset)
np.save('2016NW4647_total_cibles_6min_10h_6min.npy', cibles)

                   

        

        