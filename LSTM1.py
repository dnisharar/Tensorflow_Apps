#!/usr/bin/python3
import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import math
#import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers import Bidirectional
from keras.layers import Dropout

#from keras.models import *
#from keras.layers import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy Tensorflow warnings
warnings.filterwarnings("ignore") #Hide messy numpy warnings
np.random.seed(7)

##PART I
'''
dataframe = pd.read_csv('airline.csv',usecols=[1],engine = 'python', skipfooter = 3)

df = np.array(dataframe)
print(df.shape)
print(df[0:10,:])

'''


print("=========================================================================================================")

## PART II

data1 = pd.read_csv('~/Tensorflow/DAQ/NDX.csv')
data2 = np.array(data1)
#print("============================ data2 SHAPE ================")
#print(data2.shape)
#data2 = data2[1000:,:]
#print(data1.head())
#print(data1.tail())
print(data2[(len(data2)-10):(len(data2)+1),1])

look_back = 15
window = 3
forward = 0

def create_dataset(data2, look_back):
        X,y = [],[]
	for i in range(len(data2)-look_back):
		a = data2[i:(i+look_back), 1]
		X.append(a)
		y.append(data2[i + look_back, 1])
	return np.array(X), np.array(y)

X,y = create_dataset(data2,look_back)

print("=================== SHAPE OF X ===================================")
print(X.shape)
print("==================== TAIL OF X ====================================")
print(X[(len(X)-10):(len(X)+1),:])
print("================== SHAPE OF y =====================================")
print(y.shape)
print("==================== TAIL OF y ====================================")
print(y[(len(y)-10):(len(y)+1)])

X_train = X[:(len(X)-window),:]
print("=================== SHAPE OF X_train ==============================")
print(X_train.shape)
y_train = y[:(len(y)-window)]
print("=================== SHAPE OF y_train ==============================")
print(y_train.shape)
y_test = y[(len(y)-window):]
yhat = np.ones((len(y_test)),dtype=float)
print("================== SHAPE OF y_test ================================")
print(y_test.shape)


X_test = np.ones((window + forward, look_back),dtype = np.float32)
X_test = X_test.reshape(len(X_test),1,look_back)
#X_test[0,0:look_back] = X[(len(X)-window -1),1:]
#X_test[0,look_back] = y[len(X)-window-1]
print("================== SHAPE OF X_test ================================")
print(X_test.shape)
print(X_test[0,:,:])
#print(y_test)


X_train = X_train.reshape(len(X_train),1, look_back)
y_train = y_train.reshape(len(y_train),1,1)
X_test = X_test.reshape(len(X_test),1,look_back)
X_test[0,:,:] = X[(len(X)-window),1:]

## LSTM


print("============================================================")
n_batch = 1
n_epoch = 5

model = Sequential()
#model.add(LSTM(32, input_shape=(1,look_back), return_sequences = True, activation ='relu'))
model.add(LSTM(32, batch_input_shape = (n_batch,1,look_back), return_sequences = True, stateful = True, activation = 'relu'))
#model.add(Dropout(0.2))
#model.add(LSTM(100, return_sequences = True,shuffle = False))
#model.add(Dense(100,activation='relu'))
model.add(Dense(1,activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')

#model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=2)
for i in range(n_epoch):
    model.fit(X_train, y_train, epochs=1,batch_size = n_batch, verbose =1, shuffle =False)
    model.reset_states()

for i in range(len(X_test)):
    inp = X_test[i,:,:]
    inp = inp.reshape(1,1,look_back)
    yhat[i] = model.predict(inp, batch_size =1)

print(yhat)
print(y_test - yhat)

#trainPredict = model.predict(X_train)
print("===============PREDICTION======================================")
#print(trainPredict)
print("=================DIFFERENCE =================================")
#print(trainPredict - y_train)

#pd.DataFrame(trainPredict).to_csv("train_predicted.csv") 
#pd.DataFrame(y_test).to_csv("y_test_data.csv")

'''

testPredict = model.predict(X_test[1,0,1])

print(trainPredict-y_train)
print(testPredict)



X_test[0,:] = X[(len(X)- window - 1),:]
y_LSTM =[]
y_LSTM[0]= model.predict(X[(len(X)-window-1),:])
for k in range(1,len(y)):
    y_LSTM[k] = model.predict(X_test[(k-1),:])
    X_test[k,0:(window-1)] = X_test[(k-1),0:(window-1)]
    X_test[k,window] = y_LSTM[k]



'''

## GRU



## SimpleRNN



## BIDIRECTIONAL SimpleRNN
'''
input_tensor = input((3,1))
print(input_tensor)
X = input_tensor

for i in range(2):
    X = Bidirectional(SimpleRNN(32*2**i,return_sequences = True))(X)
    X = Activation('relu')(X)
print(X)


X = Bidirectional(SimpleRNN(512,return_sequences= False))(X)
X = Activation('relu')(X)
X = Dense(1,activation = 'linear')(X)
X = PReLU(alpha_initializer ='zeros', alpha_regularizer = None, alpha_constraint = None,shared_axes = None )(X)

model = Model(input = input_tensor, output = X)
Adamax = Adam(lr = 0.000074)
model.compile(loss='mse',optimizer = 'Adamax')

#from keras.utils.vis_utils import plot_model as plot
#from IPython.display import Image

#plot(model,to_file ='model.png',show_shapes =True)
#Image('model.png')
#plt.show()
model.fit(X_train,y_train,batch_size=100,nb_epoch =200)
predicted = model.predict(X_test)
predicted = np.reshape(predicted, (predicted.size,))
var = vairence(predicted, y_test)
#print('plotting Results')

#import matplotlib.pyplot as plt
#from matplotlib.legend_handler import HandlerLine2D

#line1, = plt.plot(ytest,marker='d',label ='Actual')
#line2, = plt.plot(predicted, marker='o',label ='Predicted')

#plt.legend(handler_map = {line1:HandlerLine2D(numpoints =4)})
#plt.show()

'''




## BIDIRECTIONAL LSTM










## BIDIRECTIONAL GRU



## LSTM BASIC
'''
data = [[i for i in range(100)]]
data = np.array(data, dtype = float)
target = [[i for i in range(1,101)]]
target = np.array(target , dtype = float)

#print(data)
#print(target)

data = data.reshape(1,1,100)
target = target.reshape(1,1,100)
X_test = [i for i in range(100,200)]
X_test = np.array(X_test).reshape(1,1,100)
y_test = [i for i in range(101,201)]
y_test = np.array(y_test).reshape(1,1,100)

model = Sequential()
model.add(LSTM(100, input_shape = (1,100), return_sequences = True))
model.add(Dense(100))
model.compile(loss = 'mean_squared_error', optimizer ='adam',metrics = ['accuracy'])
model.fit(data, target, nb_epoch = 200, batch_size = 1, verbose = 2, validation_data=(X_test,y_test))

predict = model.predict(X_test)
print(predict)

'''

## LSTM ACCURACY IMPROVEMENT
'''
leng = 3
#print(range(leng))
#print(range(1,5))
data = [[i+j for j in range(leng)] for i in range(100)]
data = np.array(data, dtype = np.float32)
print(data)
target = [[i+j+1 for j in range(leng)] for i in range(1,101)]
target = np.array(target, dtype = np.float32)
print(target)

data = data.reshape(100,1,leng)/200
#print(data)
target = target.reshape(100,1,leng)/200
#print(target)

model = Sequential()
model.add(LSTM(leng, input_shape = (1,leng), return_sequences = True, activation = 'sigmoid'))
model.add(LSTM(leng, input_shape = (1,leng), return_sequences = True, activation = 'sigmoid'))
model.add(LSTM(leng, input_shape = (1,leng), return_sequences = True, activation = 'sigmoid'))
model.add(LSTM(leng, input_shape = (1,leng), return_sequences = True, activation = 'sigmoid'))
model.add(LSTM(leng, input_shape = (1,leng), return_sequences = True, activation = 'sigmoid'))
model.add(LSTM(leng, input_shape = (1,leng), return_sequences = True, activation = 'sigmoid'))
model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
model.fit(data, target, nb_epoch = 200, batch_size = 50, validation_data =(data, target))

predict = model.predict(data)
print(predict*200)

'''


### EXAMPLE
'''
data = [0,0,0,0,0,0,0,0,0,2,1]
data = np.array(data, dtype = float)
target = [0,0,0,0,0,0,0,0,2,1,0]
target = np.array(target, dtype=float)

print(data)
print(target)

data = data.reshape((1,1,11)) # single batch, 1 time steps, 11 dimensions
target = target.reshape((-1,11)) # corresponds to shape (None, 11)

print(data)
print(target)

# Build Model
model = Sequential()  
model.add(LSTM(11, input_shape=(1, 11), unroll=True))
model.add(Dense(11))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(data, target, nb_epoch=100, batch_size=1, verbose=2)

# Do the output values match the target values?
predict = model.predict(data)
print(repr(data))
print(repr(predict))

####### more dimensions

# Input sequence
wholeSequence = [[0,0,0,0,0,0,0,0,0,2,1],
                 [0,0,0,0,0,0,0,0,2,1,0],
                 [0,0,0,0,0,0,0,2,1,0,0],
                 [0,0,0,0,0,0,2,1,0,0,0],
                 [0,0,0,0,0,2,1,0,0,0,0],
                 [0,0,0,0,2,1,0,0,0,0,0],
                 [0,0,0,2,1,0,0,0,0,0,0],
                 [0,0,2,1,0,0,0,0,0,0,0],
                 [0,2,1,0,0,0,0,0,0,0,0],
                 [2,1,0,0,0,0,0,0,0,0,0]]

# Preprocess Data:
wholeSequence = np.array(wholeSequence, dtype=float) # Convert to NP array.
data = wholeSequence[:-1] # all but last
target = wholeSequence[1:] # all but first

# Reshape training data for Keras LSTM model
# The training data needs to be (batchIndex, timeStepIndex, dimentionIndex)
# Single batch, 9 time steps, 11 dimentions
data = data.reshape((1, 9, 11))
target = target.reshape((1, 9, 11))

# Build Model
model = Sequential()  
model.add(LSTM(11, input_shape=(9, 11), unroll=True, return_sequences=True))
model.add(Dense(11))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(data, target, nb_epoch=2000, batch_size=1, verbose=2)

pred = model.predict(data)

print(pred)
'''


## CONVOLUTION NN


