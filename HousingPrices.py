#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import math
import pandas as pd
import theano as T
import datetime

from keras.models import *
from keras.layers import *

from random import random, randint

t = datetime.datetime.time(datetime.datetime.now())

data = pd.read_csv("housing.csv", delim_whitespace = True, header = None)
data1 = pd.read_csv('~/Tensorflow/DAQ/NDX2.csv')
#data2 = pd.read_csv('~/Tensorflow/DAQ/SPX.csv')
#data3 = pd.read_csv('~/Tensorflow/DAQ/RUT.csv')
print(data1.head())
print(data1.tail)
data1 = np.array(data1)


forward = 10
future = 5
lag = 5
#n1 = len(data1) - forward
X = np.array(data1)
#X = X[len(X)-2050:,1]
X = X[:,1]
n1 = len(X) - forward

def to_supervised(data, lag):
    df = pd.DataFrame(data)
    columns = [df.shift(lag +1 -i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis = 1)
    df.dropna(inplace = True)
    return df

df = to_supervised(X,lag)
df = np.array(df)
#print(df[:10,:])
print(df[(len(df)-10):,:])


train, test = df[:n1,:], df[n1:,:]
print(train.shape)
print(test.shape)

X_train ,y_train = train[:,:-1], train[:,-1]
X_test = np.zeros((forward+future,lag), dtype = float)
y_test = test[:,-1]



print(X_train.shape)
print(X_test.shape)


#X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])

print(X_test.shape)
print(X_test)
#X_test[0,:,:] = df[(n1-1),1:]
X_test[0,:] = df[(n1-1),1:]
print(X_test)

batch_size = 1
neurons = 2
bi_neurons = 2
repeats = 2
nb_epochs = 2

'''

## mlp
def mlp_model(train, batch_size, nb_epoch, neurons):
    X,y = train[:,:-1], train[:,-1]
    X = X.reshape(X.shape[0],1,X.reshape[1])
    model = Sequential()
    model.add(Dense(neurons,input_dim = (1,X.shape[2]),init = 'normal', activation = 'relu'))
    model.add(Dense(neurons, init = 'normal', activation = 'relu'))
    model.add(Dense(neurons, init = 'normal', activation = 'relu'))
    model.add(Dense(1, activation = 'linear'))
    model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
    for i in range(nb_epoch):
        model.fit(X,y, epochs = 1, batch_size = batch_size, verbose = 2, shuffle = False)
        model.reset_states()
    return model



## simple
def simple_model(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(SimpleRNN(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True, init = 'normal',return_sequences = True, activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(neurons, return_sequences = True, init = 'normal', activation = 'relu'))
        model.add(SimpleRNN(neurons, return_sequences = False, init = 'normal', activation = 'relu'))
	model.add(Dense(1, activation = 'linear'))
	model.compile(loss='mse', optimizer='adam', metrics = ['acc'])
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
		model.reset_states()
	return model

'''

##lstm
def lstm_model(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True, init = 'normal',return_sequences = True, activation = 'relu' ))
	model.add(LSTM(neurons, return_sequences = True, init = 'normal', activation = 'relu'))
        model.add(LSTM(neurons, return_sequences = False, init = 'normal', activation = 'relu'))
        model.add(Dense(1, activation = 'linear'))
	model.compile(loss='mse', optimizer='adam', metrics = ['acc'])
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
		model.reset_states()
	return model


'''

##gru
def gru_model(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(GRU(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True, init = 'normal', return_sequences = True, activation = 'relu'))
        model.add(GRU(neurons, return_sequences = True, init = 'normal', activation = 'relu'))
        model.add(GRU(neurons, return_sequences = False, init = 'normal',activation = 'relu' ))
	model.add(Dense(1, activation = 'linear'))
	model.compile(loss='mse', optimizer='adam', metrics = ['acc'])
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
		model.reset_states()
	return model




##bi_simple
def bi_simple_model(train, batch_size, nb_epoch, bi_neurons, mode):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(Bidirectional(SimpleRNN(neurons, return_sequences = True, stateful = True, init = 'normal', activation = 'relu'), input_shape =(batch_size, X.shape[2]), merge_mode = mode))
	model.add(Bidirectional(SimpleRNN(neurons, return_sequences= True,init = 'normal', activation = 'relu'), merge_mode = mode))
        model.add(Bidirectional(SimpleRNN(neurons, return_sequences = False, init = 'normal', activation = 'relu'), merge_mode = mode))
        model.add(Dense(1, activation = 'linear'))
	model.compile(loss='mse', optimizer='adam', metrics = ['acc'])
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model




##bi_lstm
def bi_lstm_model(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model




##bi_gru
def bi_gru_model(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model


'''


def forecast_models(model, batch_size, row):
    X = row #[0:-1]
    X = X.reshape(1,1,len(X))
    yhat = model.predict(X, batch_size = batch_size)
    return yhat[0,0]


#mlp_NN = mlp_model(train, batch_size, nb_epochs, neurons)
#simple_RNN = simple_model(train, batch_size, nb_epochs, neurons)
lstm_RNN = lstm_model(train, batch_size, nb_epochs, neurons)
#gru_RNN = gru_model(train, batch_size, nb_epochs, neurons)

#test[0,:-1] = X_test[0,:]
#test[0,:-1] = X_test[0,:]

def simulated_uni(model, train, batch_size, nb_epochs, neurons):
    n1 = len(X_test)
    n2 = repeats
    predictions1 = np.zeros((n1,n2), dtype = float)
    for r in range(repeats):
        #my_model = model(train, batch_size, nb_epochs, neurons)
        #train_reshaped = train[:,:-1].reshape(X_train.shape[0],1,X_train.shape[1])
        #my_model.predict(train_reshaped, batch_size =1)
        predictions2 = list()
        for i in range(len(X_test)):
            #X = test[i,:-1] #,y =  test[i,-1]
            if(i==0):
               y = forecast_models(model, batch_size, X_test[i,:])
               X_test[i+1,:-1] = X_test[i,1:]
               X_test[i+1,-1] = y
               predictions2.append(y)
            else:
               y = forecast_models(model,batch_size,X_test[i-1,:])
               X_test[i,:-1] = X_test[i-1,1:]
               X_test[i,-1] = y
               #yhat = forecast_models(my_model,1,X)
               #predictions2.append(yhat)
               predictions2.append(y)
        predictions1[:,r] = predictions2
    return np.mean(predictions1, axis = 1)


##pred

'''

y_simple = []
y_lstm = []
y_gru = []
y_bisimple = []
y_bilstm = []
y_bigru = []


y_simple[0] = simple_model.predict(X_test[0,:,:], batch_size =1)


y_lstm[0] = lstm.predict()
y_gru[0]  = gru.predict()
y_bisimple[0] = bi_simple.predict()
y_bilstm[0] = bi_lstm.predict()
y_bigru[0] = bi_gru.predict()


for k in range(1,len(X_test)):
    y_simple[k] = simple_model.predict(X_test[k,:,:],batch_size =1)
    X_test[k,:,:-1] = X_test[k-1,:,1:]
    X_test[k,:,-1] = y_simple[k]

'''
#t = datetime.datetime.time(datetime.datetime.now())
print(datetime.datetime.time(datetime.datetime.now()))
print(y_test)
#print(y_simple)
print("===================== mlp =============================")
#print(simulated_uni(mlp_NN, train, batch_size, nb_epochs, neurons))
print("====================simple ============================")
#print(simulated_uni(simple_RNN, train,batch_size, nb_epochs, neurons))
print("=======================lstm =============================")
print(simulated_uni(lstm_RNN, train, batch_size, nb_epochs, neurons))
print("========================= gru ===============================")
#print(simulated_uni(gru_RNN, train, batch_size, nb_epochs, neurons))
print("===================== true ================================")
print(y_test)

print(t)
print(datetime.datetime.time(datetime.datetime.now()))
