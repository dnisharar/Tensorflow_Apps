import tensorflow as tf
import numpy as np
import math
import pandas as pd
import theano as TH

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import *
from keras.layers import *

from random import random, randint
from numpy import cumsum

from keras.layers import Flatten, Activation, Convolution2D, MaxPooling2D



#generate a sequence of real values between 0 and 1

def generate_sequence(length):
    return np.array([i/float(length) for i in range(length)])

data = generate_sequence(100)
#print(data)

def generate_sequence2(length):
    return [i for i in range(length)]

seq1 = generate_sequence2(5)
#print(seq1)

def generate_randint(length):
    return [randint(0,99) for _ in range(length)]

seq2 = generate_randint(10)
#print(seq2)

def generate_random(n_timesteps):
    X = np.array([random() for _ in range(n_timesteps)])
    limit = n_timesteps/4
    y = np.array([0 if x < limit else 1 for x in cumsum(X)])
    return X,y

X,y = generate_random(10)
#print(X)
#print(y)

X = X.reshape(10,1,1)
y = y.reshape(10,1,1)

#print(X)
#print(y)


'''
## MLP

model = Sequential()
model.add(Dense(100,input_dim = 1,activation='relu',init = 'normal'))
model.add(Dropout(0.2))
model.add(Dense(100, activation ='relu', init = 'normal'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu', init = 'normal'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear', init = 'normal'))

model.compile(loss = 'mse', optimizer ='adam',metrics = ['accuracy'])

model.fit(X,y,nb_epoch =200, batch_size = 2, verbose = 2)
pred = model.predict(X)
print(pred-y)
scores = model.evaluate(X,y,verbose = 0)
print(scores)
print("Model Accuracy: %.2f%%" %(scores[1]*100))
print("Model Loss: %.2f%%" %(scores[0]*100))
print(model.summary())
#print(model.get_config())
#from keras.utils.visualize_util import plot
#plot(model, to_file = 'model.png')
'''



'''
## LSTM
model = Sequential()
model.add(LSTM(32 , input_shape=(1,1), return_sequences = True, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))
model.compile(loss ='mse',optimizer ='adam') #,metrics ='acc')
model.fit(X,y,nb_epoch = 20, batch_size = 1, verbose = 2)

print("=========== LSTM pred ==============================================")
pred = model.predict(X[1,0,0])
print(pred)

print("=========== LSTM difference ========================================")
print(pred-X)

## GRU
model_GRU = Sequential()
model_GRU.add(GRU(32, input_shape =(1,1), return_sequences = True, activation = 'relu'))
model_GRU.add(Dense(1, activation = 'relu'))
model_GRU.compile(loss = 'mse', optimizer = 'adam')
model_GRU.fit(X,y, nb_epoch = 20, batch_size = 1, verbose = 2)

print("=========== GRU pred ==============================================")
pred_GRU = model_GRU.predict(X)
print(pred_GRU)
print("=========== GRU difference ==========================================")
print(X - pred_GRU)


## SimpleRNN

model_SimpleRNN = Sequential()
model_SimpleRNN.add(SimpleRNN(32, input_shape = (1,1), return_sequences = True, activation = 'relu'))
model_SimpleRNN.add(Dense(1, activation = 'relu'))
model_SimpleRNN.compile(loss = 'mse', optimizer = 'adam')
model_SimpleRNN.fit(X,y, nb_epoch = 20, batch_size = 1, verbose = 2)

print("========== SimpleRNN pred===========================================")
pred_SimpleRNN = model_SimpleRNN.predict(X)
print(pred_SimpleRNN)
print("========== SimpleRNN difference ====================================")
print(X - pred_SimpleRNN)

'''

###################  BATCH/ONLINE PROCESSING ################################### 

'''
length = 10
#sequence =[random()+ i  for i in range(length)]
sequence = [i/float(length) for i in range(length)]
print(sequence)
print(np.array(sequence))
#create X/y pairs
df = pd.DataFrame(sequence)
print(df)
df = pd.concat([df.shift(3),df.shift(2),df.shift(1),df], axis =1)
print(df)
df.dropna(inplace =True)
print(df)


#convert to LSTM friendly format
print("====================== convert=================================")
values = df.values
print(values)
X,y = values[:,0:3], values[:,3]
print(X)
print(y)
X = X.reshape(len(X),1,3)
print(X)
print(X.shape, y.shape)
print(X.shape[0],X.shape[1],X.shape[2])
print(X[3,:,:])
print(y[3])


#configure network
n_batch = len(X)
n_epoch = 200
n_neurons = 10
#design network
model = Sequential()
model.add(LSTM(n_neurons, batch_input_shape = (n_batch, X.shape[1], X.shape[2]),return_sequences = False,  stateful = True, activation ='relu'))
model.add(Dense(1))
model.compile(loss ='mse', optimizer = 'adam')

#fit network

for i in range(n_epoch):
    model.fit(X,y, epochs =1, batch_size = n_batch, verbose =2, shuffle = False)
    model.reset_states()

'''

'''
#online forecast
for i in range(len(X)):
    testX, testy = X[i,:,:],y[i]
    testX = testX.reshape(1,1,3)
    yhat = model.predict(testX,batch_size =1)
    print('>Expected =%.1f, predicted = %.1f' %(testy, yhat))

'''


##### COPY WEIGHTS 
'''
n_batch =1
new_model =  Sequential()
new_model.add(LSTM(n_neurons, batch_input_shape =(n_batch, X.shape[1], X.shape[2]), return_sequences = False, stateful = True, activation = 'relu'))
new_model.add(Dense(1))
#copy weights
old_weights = model.get_weights()
new_model.set_weights(old_weights)
#new_model.compile(loss = 'mse', optimizer = 'adam')
#online forecast
for i in range(len(X)):
    testX, testy = X[i,:,:],y[i]
    testX = testX.reshape(1,1,3)
    yhat = new_model.predict(testX, batch_size = n_batch)
    print('>Expected = %.1f, Predicted = %.1f' %(testy,yhat))

print(model.summary())
print("=======================================")
print(new_model.summary())

'''

############################### MEMORY DEMO WITH LSTM ##################################################


'''
def encode(pattern, n_unique):
    encode = list()
    for _ in pattern:
        row = [0.0 for x in range(n_unique)]
        row[_] = 1.0
        encode.append(row)
    return encode


def to_xy_pairs(encoded):
    X,y = list(), list()
    for i in range(1, len(encoded)):
        X.append(encoded[i-1])
        y.append(encoded[i])
    return X,y


seq1 = [3,0,1,2,3]
encoded1 = encode(seq1,5)
for _ in encoded1:
    print(_)

print("===================================")
seq2 = [4,0,1,2,4]
encoded2 = encode(seq2,5)
for _ in encoded2:
    print(_)

print("===================================")
X,y = to_xy_pairs(encoded1)
for i in range(len(X)):
    print(X[i],y[i])

print("===================================")
X2,y2 = to_xy_pairs(encoded2)
for _ in range(len(X2)):
    print(X2[_],y2[_])



#reshape data

df = pd.DataFrame(X)
print("======= reshaping =============")
print(df)
values = df.values
print(values)
ar = values.reshape(4,1,5)
print(ar)


def to_lstm_dataset(sequence, n_unique):
    #one hot encoding
    encoded = encode(sequence, n_unique)
    #convert to in/out patterns
    X,y = to_xy_pairs(encoded)
    #convert to LSTM friendly data
    dfX, dfy = pd.DataFrame(X), pd.DataFrame(y)
    lstmX = dfX.values
    lstmX = lstmX.reshape(lstmX.shape[0],1,lstmX.shape[1])
    lstmY = dfy.values
    lstmY = lstmY.reshape(lstmY.shape[0],1,lstmY.shape[1])
    return lstmX, lstmY

seq1 = [3,0,1,2,3]
seq2 = [4,0,1,2,4]
n_unique = len(set(seq1 + seq2))
print(seq1+seq2)
print(set(seq1+seq2))

seq1X, seq1Y = to_lstm_dataset(seq1, n_unique)
seq2X, seq2Y = to_lstm_dataset(seq2, n_unique)

print("=============== seqs ======================================")
print(seq1X)
print(seq1Y)
print(seq2X)
print(seq2Y)


#model configuration
model = Sequential()
model.add(LSTM(20, batch_input_shape = (1,1,5), return_sequences = True, stateful = True, activation = 'relu'))
model.add(Dense(5, activation = 'linear'))
model.compile(loss = 'mse', optimizer = 'adam')


#model training
for i in range(250):
    model.fit(seq1X, seq1Y, epochs =1, batch_size =1, verbose =1, shuffle = False)
    model.reset_states()
    model.fit(seq2X, seq2Y, epochs =1, batch_size =1, verbose =0, shuffle = False)
    model.reset_states()

#model Evaluation

result = model.predict_classes(seq1X, batch_size =1, verbose =0)
print(result)
result2 = model.predict_classes(seq2X, batch_size = 1, verbose = 1)

print("========================================================")
print(result)
print(result2)

'''


################# ECHO RANDOM INTEGERS ###################################
length = 25
n_unique = 100
top = 99
#generate a sequence of random integers
def generate_sequence(length):
    return [randint(0,top) for _ in range(length)]

seq1 = generate_sequence(length)
print(seq1)

#one hot encode the sequence
def one_hot_encode(sequence, n_unique):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return np.array(encoding)

encode1 = one_hot_encode(seq1,n_unique)
print(encode1)

#decode any encoded sequence
def one_hot_decode(encoded_seq):
    return [np.argmax(vector) for vector in encoded_seq]

print(one_hot_decode(encode1))

#generate data for lstm
def generate_data(length, n_unique):
    sequence = generate_sequence(length)
    encoded = one_hot_encode(sequence, n_unique)
    X = encoded.reshape(encoded.shape[0],1,encoded.shape[1])
    return X,encoded

X,encode = generate_data(length, n_unique)
print(X)

#define the model
model = Sequential()
model.add(LSTM(15, input_shape=(1,n_unique)))
model.add(Dropout(0.2))
model.add(Dense(n_unique, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc'])

#fit model
for i in range(50):
    X,y = generate_data(length, n_unique)
    model.fit(X,y,epochs = 1, batch_size =1, verbose = 2)

#evaluate model on new data
X,y = generate_data(length, n_unique)
yhat = model.predict(X)
print('Expected: %s' %one_hot_decode(y))
print('Predicted: %s' %one_hot_decode(yhat))

# echo lag without context
print("========================= ECHO WITHOUT CONTEXT =====================")

#updated generate data for the lstm function
def generate_data2(length, n_unique):
    sequence = generate_sequence(length)
    encoded = one_hot_encode(sequence, n_unique)
    #drop first value from X
    print(encoded)
    print("===========================")
    X = encoded[1:,:]
    print(X)
    #convert to 3D for input
    print("============================")
    X = X.reshape(X.shape[0],1,X.shape[1])
    print(X)
    #drop last value from y
    y = encoded[:-1,:]
    y = y.reshape(y.shape[0],1,y.shape[1])
    return X,y

X,y =generate_data2(length,n_unique)
print("===================================")
print(X)
print(y)

for i in range(len(X)):
    a,b = np.argmax(X[i,0]), np.argmax(y[i])
    print(a,b)

#configure the model
batchsize = 6
model = Sequential()
model.add(LSTM(100, batch_input_shape = (batchsize,1,n_unique), stateful = True, return_sequences = True, activation ='relu', init = 'normal'))
model.add(Dropout(0.2))
model.add(Dense(50, activation = 'relu', init = 'normal'))
model.add(Dropout(0.2))
model.add(Dense(n_unique, activation = 'linear'))
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])

#fit model
for i in range(50):
    X,y = generate_data2(length, n_unique)
    model.fit(X,y,epochs = 1, batch_size = batchsize, verbose = 2, shuffle = False)
    model.reset_states()

X,y = generate_data2(length, n_unique)
yhat = model.predict(X, batch_size = batchsize)
print('Expected:  %s' % one_hot_decode(y))
print('Predicted: %s' % one_hot_decode(yhat))

## echo lag observation

print("======================== ECHO LAG ==================================")
#generate for LSTM again
def generate_data3(length, n_unique):
    sequence = generate_sequence(length)
    encoded = one_hot_encode(sequence, n_unique)
    df = pd.DataFrame(encoded)
    df = pd.concat([df.shift(4),df.shift(3),df.shift(2),df.shift(1),df], axis =1)
    values = df.values
    values = values[5:,:]
    X = values.reshape(len(values),5,n_unique)
    #drop last value from y
    y = encoded[4:-1,:]
    #y = y.reshape(y.shape[0],5,n_unique)
    print(X.shape, y.shape)
    return X,y

X,y = generate_data3(length, n_unique)
print(X)
print(y)

#test data generator
for i in range(len(X)):
    a,b,c,d,e,f = np.argmax(X[i,0]),np.argmax(X[i,1]),np.argmax(X[i,2]),np.argmax(X[i,3]),np.argmax(X[i,4]),np.argmax(y[i])
    print(a,b,c,d,e,f)

#define model
batch = 5
model = Sequential()
model.add(LSTM(100,batch_input_shape = (batch, 5, n_unique),stateful = True, return_sequences = True, activation ='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(n_unique, activation ='linear'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

#fit model
for i in range(50):
    X,y = generate_data3(length, n_unique)
    model.fit(X,y,epochs =1, batch_size = batch, verbose = 2, shuffle = False)
    model.reset_states()

#evaluate model
X,y = generate_data3(length, n_unique)
yhat = model.predict(X, batch_size = batch)
print('Expected : %s' % one_hot_decode(y))
print('Predicted: %s' % one_hot_decode(yhat))
