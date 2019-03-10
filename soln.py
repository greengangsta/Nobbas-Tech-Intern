### Importing the libraries ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 


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

# Geo-coding for the latitude and longitude sadly I don't have an api key
"""
from geopy import geocoders
api_key = 'AIzaSyBXkATWIrQyNX6T-VRa2gRmC9dJRoqzss0'
g = geocoders.GoogleV3(api_key=api_key) 
location = Xt[:,2] +', '+ Xt[:,0] +  ', ' + Xt[:,1] 
for loc in location :
 try:
    place, (lat, lng) = g.geocode(loc)
 except ValueError as error_message:
    print("Error: geocode failed on input %s with message %s" % (location, error_message))
 print (place, lat, lng)

"""
imputer = imputer.fit(Xt[:,10:12])
Xt[:,10:12]= imputer.transform(Xt[:,10:12])
 








