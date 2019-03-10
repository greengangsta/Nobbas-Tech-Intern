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
y[:]=le.fit_transform(y[:])
y=y.astype(int)
prices = X.iloc[:,13].values

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

# Geo-coding for the latitude and longitude sadly I don't have an api key hence i'm imputing the latitude and longitude
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

# Label Encoding
Xt[:,0]=  le.fit_transform(Xt[:,0])
Xt[:,1]=  le.fit_transform(Xt[:,1])
Xt[:,2]=  le.fit_transform(Xt[:,2])
Xt[:,3]=  le.fit_transform(Xt[:,3])
Xt[:,4]=  le.fit_transform(Xt[:,4])
Xt[:,5]=  le.fit_transform(Xt[:,5])
Xt[:,9]=  le.fit_transform(Xt[:,9])
Xt[:,14]=  le.fit_transform(Xt[:,14])
Xt[:,17]=  le.fit_transform(Xt[:,17])
Xt[:,18]=  le.fit_transform(Xt[:,18])
Xt[:,19]=  le.fit_transform(Xt[:,19])
Xt[:,20]=  le.fit_transform(Xt[:,20])
Xt[:,21]=  le.fit_transform(Xt[:,21])

#Dropping the description column 
Xt = np.delete(Xt,8,1)
Xpt = np.delete(Xt,11,1)
Xt = np.delete(Xt,23,1)
print(Xt[1,:])
print(Xpt[1,:])


# onehot encoding some of the important categorical features
onehotencoder = OneHotEncoder(categorical_features =[1,4,14,17,18,19])
Xt= onehotencoder.fit_transform(Xt).toarray()
onehotencoder = OneHotEncoder(categorical_features =[1,4,13,16,17,18])
Xpt = onehotencoder.fit_transform(Xpt).toarray()


### Building the machine learning models ###

# Splitting the dataset into train and test data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size = 0.2, random_state = 0)
Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xpt, prices, test_size = 0.2, random_state = 0)


# Scaling the dataset
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
Xp_train = sc_X.fit_transform(Xp_train)
Xp_test = sc_X.transform(Xp_test)


# Training and prediction for the grade classification
from sklearn.svm import SVC
classifier1 = SVC(kernel='linear')
classifier1.fit(X_train,y_train)
y_pred = classifier1.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(f1_score(y_test, y_pred, average="macro"))
print(precision_score(y_test, y_pred, average="macro"))
print(recall_score(y_test, y_pred, average="macro")) 
print(accuracy_score(y_test, y_pred))

 
# Regression model for price prediction
yp_train = yp_train.astype(int)
yp_test = yp_test.astype(int)
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(Xp_train,yp_train)
yp_pred = regressor.predict(Xp_test)

# Calculating the rmse
from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(yp_test, yp_pred))







