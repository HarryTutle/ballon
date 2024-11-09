# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:01:25 2024

@author: Harry
"""

import numpy as np
import pandas  as pd
import glob
from sklearn.model_selection import train_test_split

mean_tenseur=[42259671.94, 4595.35, 237.56, 142.38, 187.01, 3.84, 74.49, 286.58, 1017.05, 6.52]
std_tenseur=[25851787.75, 253.68, 385.51, 179.13, 111.68, 2.90,  18.05, 7.29, 833.26, 3.45]

data=np.load('D:/station meteo 2024/2016SE_total45504600_dataset_1h_10h_6min.npy')
cible=np.load('D:/station meteo 2024/2016SE_total45504600_cibles_1h_10h_6min.npy')

#data=data[:100]
#cible=cible[:100]

''' standardisation des données'''

dim_1_data=data.shape[0]
dim_2_data=data.shape[1]
dim_3_data=data.shape[2]


data=(data-mean_tenseur) / std_tenseur    
   
#data2=(data*std_tenseur)+mean_tenseur

def label_direction(val):
    
    if val==0:
        
        return 0
    
    elif val==45:
        
        return 1
    
    elif val==90:
        
        return 2
    
    elif val==135:
        
        return 3
    
    elif val==180:
        
        return 4
    
    elif val==225:
        
        return 5
    
    elif val==270:
        
        return 6
    
    elif val==315:
        
        return 7
    
    else:
        
        return 'nan'
    
      
    
''' configuration des données pour le pipeline avec tf.data'''

data=data.reshape(dim_1_data, dim_2_data*dim_3_data)

pack=np.concatenate([data, cible], axis=1)

pack=pd.DataFrame(pack)
pack[1000]=pack[1000].apply(lambda x: label_direction(x))

train, test = train_test_split(pack, test_size=0.3, random_state=42)
test, valid = train_test_split(test, test_size=0.5, random_state=42)

train.to_csv("D:/station meteo 2024/lium_data/train/2016SE_45504600_1h_10h_6min_tr.csv")
test.to_csv("D:/station meteo 2024/lium_data/test/2016SE_45504600_1h_10h_6min_te.csv")
valid.to_csv("D:/station meteo 2024/lium_data/valid/2016SE_45504600_1h_10h_6min_va.csv")




"""

data=data[:100000]
cible=cible[:100000]

new_cible=np.expand_dims(cible, 1)
new_cible=np.repeat(new_cible, 100, axis=1)
new_cible=np.repeat(new_cible, 5, axis=2)

pack=np.concatenate([data, new_cible], axis=2)
pack=pack[:,:,[i for i in range(0, 11)]+[-1]]

dim_1=pack.shape[0]
dim_2=pack.shape[1]
dim_3=pack.shape[2]

pack=pack.reshape(dim_1, dim_2*dim_3)
pack=pd.DataFrame(pack)

train, test = train_test_split(pack, test_size=0.3, random_state=42)
test, valid = train_test_split(test, test_size=0.5, random_state=42)



debug=pack.reshape(dim_1, dim_2, dim_3)



print(np.sum(data-debug[:,:,:-2]))
print(np.sum(cible-debug[:,0,-2:]))





np.save("C:/Users/Harry/Documents/station meteo 2024/lium_data/train/2017SE_total43504400_1h_10h_6min.npy" , train)
np.save("C:/Users/Harry/Documents/station meteo 2024/lium_data/test/2017SE_total43504400_1h_10h_6min.npy" , test)
np.save("C:/Users/Harry/Documents/station meteo 2024/lium_data/valid/2017SE_total43504400_1h_10h_6min.npy" , valid)

"""