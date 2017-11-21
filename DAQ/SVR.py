#!/usr/bin/python2
"""
Created on Sun Jul 16 21:14:28 2017

@author: T
"""
import tensorflow as tf
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasClassifier
#from keras.wrappers.scikit_learn import GridsearchCV
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn import grid_search, linear_model
from sklearn import preprocessing
import matplotlib.pyplot as plt

from sklearn.neural_network import  MLPRegressor

import spark
#from spark import spark_sklearn
#from spark_sklearn import GridSearchCV, SVR


window = 10

df = pd.read_csv("NDX.csv")
print(df.head())
print("===================")
print(df.tail())


df = df[['DATE','CLOSE']]
df1 = np.array(df)


df2 = np.ones((len(df1)-window,window),dtype = np.float64)
for k in range(window, len(df2)):
    for j in range(window):
        df2[k,j] = df1[k+j,1]


print(df2[1:20,:])
print(df2[len(df2)-10:len(df2),:])
print(df.head())
print(df.tail())


df['X0'] = df['CLOSE'].shift(-window)
df['X1'] = df['CLOSE'].shift(-(window-1))
df['X2'] = df['CLOSE'].shift(-(window-2))
df['X3'] = df['CLOSE'].shift(-(window-3))
df['X4'] = df['CLOSE'].shift(-(window-4))
df['X5'] = df['CLOSE'].shift(-(window-5))
df['X6'] = df['CLOSE'].shift(-(window-6))
df['X7'] = df['CLOSE'].shift(-(window-7))
df['X8'] = df['CLOSE'].shift(-(window-8))
df['X9'] = df['CLOSE'].shift(-(window-9))
df['X10'] = df['CLOSE'].shift(-(window-10))


data = df[['DATE','CLOSE','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10']]

data = data[:-window]
le = len(data)-window
le = len(data)- 50
data = data[-le:]
print(data.head(25))
print(data.tail(25))


X = data.drop(data.columns[[0, 1, 2]], axis=1)
print(X.head(25))
print(X.tail(25))

y = np.array(data['X0'])


wid = 4
horizon2 = len(X)
horizon1 = horizon2 - wid
X_train = np.array (X)
X_train = X_train[:(horizon1+1),:]
y_train = y[:(horizon1+1)]
y_test = y[(horizon1+1):]
 
X_test = np.ones((wid,10),dtype=float) 
y_predOLS =np.ones(((wid-1),1),dtype=float) 
y_predSVR =np.ones(((wid-1),1),dtype=float) 
y_predMLP =np.ones(((wid-1),1),dtype=float) 
X_test[[0]] = X.iloc[horizon1 + 1]


''' OLS '''


OLS_model = linear_model.LinearRegression()

# Train the model using the training sets
OLS_model.fit(X_train,y_train) 
y_predOLS[0] = OLS_model.predict(X_test[[0]])

for k in range(1,(wid-1)):
    y_predOLS[k]     = OLS_model.predict(X_test[[k-1]])
    X_test[k,:8] = X_test[(k-1),:8] 
    X_test[k,9] = y_predOLS[k] 



y_test = y_test.reshape((wid-1),1) 
print(y_test - y_predOLS)



''' SVR '''

#Model Optimization 
#parameters    = {'kernel':('linear','poly', 'rbf'),'C':[1, 10, 100, 1000], 'gamma': np.logspace(-2, 1, 4,base=2),'epsilon':np.logspace(-2,1,4,base=10)} 
parameters    = {'kernel':('linear','poly', 'rbf'),'C':[1, 10,100,1000], 'gamma': np.logspace(-2, 1, 10,base=2),'epsilon':np.logspace(-2,1,10,base=10)}
SVR_model     = SVR()
grid           = GridSearchCV(SVR_model, parameters)
SVR_model = grid.fit(X_train, y_train)


#SVR_model_train = grid.fit(X, y) 
#SVR_model_optimized     = SVR(kernel = grid.best_params_["kernel"], C=grid.best_params_["C"], gamma=grid.best_params_["gamma"],epsilon = grid.best_params_["epsilon"]) 
#SVR_model_optimized.fit(X, y)

  
#y_predSVR[0]     = SVR_model_optimized.predict(X_test.iloc[[0]]) 


#SVR_model = SVR_model.fit(X_train, y_train) 

y_predSVR[0]     = SVR_model.predict(X_test[[0]]) 
 
for k in range(1,(wid-1)):
    y_predSVR[k]     = SVR_model.predict(X_test[[k-1]])
    X_test[k,:8] = X_test[(k-1),:8] 
    X_test[k,9] = y_predSVR[k] 
     
len(y_test)
len(y_predSVR)
#y_test = y_test.reshape(3,1) 

dif = y_test - y_predSVR
print(dif)









''' NEURAL NETWORK '''
'''
MLP_model = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
MLP_model.fit(X_train, y_train)

y_predMLP[0] = MLP_model.predict(X_test[[0]])

for k in range(1,(wid-1)):
    y_predMLP[k]     = MLP_model.predict(X_test[[k-1]])
    X_test[k,:8] = X_test[(k-1),:8] 
    X_test[k,9] = y_predMLP[k] 

#y_test = y_test.reshape((wid-1),1)

 
y_test - y_predOLS
y_test - y_predSVR
y_test - y_predMLP

'''



'''
t = np.linspace(1,wid,3)
plt.plot(t,y_test)
plt.show()

lines = plt.plot(t, y_test, t, y_predSVR)
plt.show()

output = pd.DataFrame({'y': y_test, 'y_SVR': y_predSVR})
output['diff'] = output['y'] - output['y_SVR']

print(output)

'''

 
''' https://www.hackerearth.com/practice/machine-learning/data-manipulation-visualisation-r-python/tutorial-data-manipulation-numpy-pandas-python/tutorial/ '''


### TENSORFLOW 
'''
print('=============================================================================================================>')

mydf2 = tf.Variable('float', [1323,13])
mydf2 = tf.Variable(df)
print(mydf2)
print(mydf2.shape)

X = mydf2[:,0:-1]
y = mydf2[:,-1]

print(X)
print(y)
'''


