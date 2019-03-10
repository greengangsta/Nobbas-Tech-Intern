### Importing the libraries ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy import geocoders 


### Loading the dataset ###

X = pd.read_csv('CSV 1.csv')
X2 = pd.read_csv('CSV 2.csv')
X3 = pd.read_csv('CSV 3.csv')


#Selecting the grade feature as the ground-truth
y = X.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
y[:]=  np.array(le.fit_transform(y[:]))

### Pre-processing the data ###

#Filling the missing values
X['stateOrProvince']=X['stateOrProvince'].fillna('U')
X['unitNumber']=X['unitNumber'].fillna('U')
X['geocodioAccuracyScore'] = X['geocodioAccuracyScore'].fillna(-1)
X['numParkingSpaces']=X['numParkingSpaces'].fillna(-1)

#Imputing the missing values with mean
Xt = X.iloc[:,1:].values
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values ='NaN',strategy = 'mean',axis = 0)
imputer = imputer.fit(Xt[:,15:17])
Xt[:,15:17] = imputer.transform(Xt[:,15:17])
imputer = imputer.fit(Xt[:,13:14])
Xt[:,13:14] = imputer.transform(Xt[:,13:14])






