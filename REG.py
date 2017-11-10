#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import math
import pandas as pd
import theano as T
import urllib
#import quandl

#from pandas.io.data import DataReader
#from pandas_datareader import data
#from datetime import datetime

from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Activation, Convolution2D, MaxPooling2D

from keras.layers import *
from keras.models import *

from random import random, randint
from pandas import datetime
#from matplotlib import pyplot

#import matplotlib.pyplot as plt

#from keras.wrappers.scikit_learn import kerasRegressor

#regression using housing data
#load dataset
dataframe = pd.read_csv("housing.csv", delim_whitespace = True, header = None)
#print(dataframe.shape)
data = dataframe.values
dataset = dataframe.values

mi = np.min(dataset, axis = 1)
Ma = np.max(dataset, axis = 1)

for i in range(dataset.shape[1]):
    dataset[:,i] = (dataset[:,i]-mi[i])/Ma[i]

#split into X and y
X = dataset[:,0:13]
y = dataset[:,13]
#print(X.shape)
#print(y.shape)

b = 499
X_train = X[0:b,:]
X_test = X[b:,:]
y_train = y[0:b]
y_test = y[b:]
#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)

## MLP
'''
model = Sequential()
model.add(Dense(1000,input_dim = 13, init = 'normal', activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, init = 'normal', activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, init = 'normal', activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, init = 'normal', activation = 'relu'))
model.add(Dense(1,init = 'normal',activation = 'linear'))
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])

model.fit(X_train, y_train, nb_epoch = 5, batch_size = 20)

pred = model.predict(X_test)
pred = pred.reshape(1,len(y_test))

y_test = (y_test + mi[dataset.shape[1]])*Ma[dataset.shape[1]]
pred = (pred + mi[dataset.shape[1]])*Ma[dataset.shape[1]]

print("=========================================")
print(data[b:,13])
print(y_test)
print(pred)
print(y_test - pred)

'''

## BIDIRECTIONAL
'''
#create a sequence classification instance
def get_sequence(n_timesteps):
    #create a sequence of random numbers in [0,1]
    X = np.array([random() for _ in range(n_timesteps)])
    #calculate cut-off value to change class values
    limit = n_timesteps/4
    #determine class outcome
    y = np.array([0 if x < limit else 1 for x in np.cumsum(X)])
    #reshape for LSTM
    X = X.reshape(1, n_timesteps,1)
    y = y.reshape(1, n_timesteps,1)
    return X,y

#n_timesteps = 10
#X,y  = get_sequence(n_timesteps)
#print(X)
#print(y)

#define LSTM model
def lstm_model(n_timesteps, backwards):
    model = Sequential()
    model.add(LSTM(500,input_shape = (n_timesteps,1), return_sequences = True, activation ='relu', go_backwards = backwards))
    model.add(LSTM(500,return_sequences = True, activation = 'relu'))
    model.add(TimeDistributed(Dense(1, activation = 'sigmoid')))
    #model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics =['acc'])
    return model


#define Bidirectional model
def bi_lstm_model(n_timesteps, mode):
    model = Sequential()
    model.add(Bidirectional(LSTM(100, return_sequences = True), input_shape = (n_timesteps,1),merge_mode = mode))
    model.add(Bidirectional(LSTM(100, return_sequences = True),merge_mode = mode))
    model.add(TimeDistributed(Dense(1, activation = 'sigmoid')))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics =['acc'])
    return model


#fit model
def train_model(model, n_timesteps):
    loss = list()
    for _ in range(10):
        X,y = get_sequence(n_timesteps)
        hist = model.fit(X,y, epochs = 1, batch_size = 1, verbose = 2)
        loss.append(hist.history['loss'][0])
    return loss


#evaluate the models
n_timesteps = 10
results = pd.DataFrame()
print(results)

#lstm forwards
model = lstm_model(n_timesteps, False)
results['lstm_forw'] = train_model(model, n_timesteps)
print(results)

#lstm backwards
model = lstm_model(n_timesteps, True)
results['lstm_back']= train_model(model, n_timesteps)
print(results)

#bidirectional concat
model = bi_lstm_model(n_timesteps, 'concat')
results['bilstm_con'] = train_model(model, n_timesteps)
print(results)

#bidirection ave
model = bi_lstm_model(n_timesteps, 'ave')
results['bilstm_ave'] = train_model(model, n_timesteps)
print(results)

#bidirectional sum
model = bi_lstm_model(n_timesteps, 'sum')
results['bilstm_sum'] = train_model(model, n_timesteps)
print(results)

#bidirectional mul
model = bi_lstm_model(n_timesteps, 'mul')
results['bilstm_mul'] =train_model(model, n_timesteps)
print(results)

'''


## UNIVARIATE TIME SERIES
'''

print("================ UNIVARIATE TIME SERIES FORECASTING ==============")

# load dataset
#def parser(x):
#	return datetime.strptime('190'+x, '%Y-%m')
#series = pd.read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
series = pd.read_csv('shampoo-sales.csv', usecols =[1], engine = 'python', skipfooter=2)
print(series)
print("======================")
X = series.values
print(X[0:10,:])
print("=====================")
print(X[(len(X)-10):len(X),:])
print("=====================")
print(X.shape)


train, test = X[0:-12],X[-12:]

#persistence forecast
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    #make predictions
    predictions.append( history[-1])
    history.append(test[i])

print(test)
print(np.array(predictions))


series = np.array(X)
#frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag):
    df = pd.DataFrame(data)
    columns = [df.shift(lag + 1 -i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis =1)
    df.fillna(0, inplace = True)
    return df

supervised = timeseries_to_supervised(X,5)
print("====================================")
print(supervised.head())
print("====================================")
print(supervised.tail())
sup = np.array(supervised)
print("====================================")
print(sup[0:12,:])
print("====================================")
print(sup[(len(sup)-12):len(sup),:])


#create a differenced series
def difference(dataset, interval =1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)

#invert differenced value
def inverse_difference(history, yhat, interval =1):
    return yhat + history[-1]

dif = difference(X,1)
print(dif)
dif2 = difference(series)
print(dif2)
print(dif - dif2)


inverted = list()
for i in range(len(dif)):
    value = inverse_difference(X, dif[i], len(X)-i)
    inverted.append(value)

inverted = np.array(inverted)
print(inverted[0:12,:])
print(X[0:12,:])


#define lstm model
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X_train, y_train = train[:,0:-1], train[:,-1]
    X_train = X_train.reshape(19,1,5)
    #y_train = y_train.reshape(22,1,1)
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape =(batch_size, X_train.shape[1],X_train.shape[2]),stateful = True, init = 'normal', activation ='relu', return_sequences = True))
    model.add(LSTM(neurons, return_sequences = True , init = 'normal', activation = 'relu'))
    model.add(LSTM(neurons, return_sequences = False, init = 'normal', activation = 'relu'))
    model.add(Dense(1))
    model.compile(loss = 'mse', optimizer = 'adam')
    for i in range(nb_epoch):
        model.fit(X_train, y_train,epochs =1, batch_size = batch_size, verbose =2, shuffle = False)
        model.reset_states()
    return model


def forecast_lstm(model, batch_size, row):
    X = row #[0:-1]
    X = X.reshape(1,1,len(X))
    yhat = model.predict(X, batch_size = batch_size)
    return yhat[0,0]

#transform for stationarity
diff_values = X #difference(X, 1)


#transform for supervised learning
supervised = timeseries_to_supervised(diff_values, 5)
supervised_values = supervised.values
supervised_values = supervised_values[5:,:]
print(supervised_values[0:12,:])



#split data into training and test set
train , test = supervised_values[0:-12,:], supervised_values[-12:,:]

print(train.shape)
print(test.shape)


#fit model
lstm_model = fit_lstm(train, 1, 10,50)
print(lstm_model.summary())


#train_reshaped = train[:,0:train.shape[1]].reshape(len(X),1,(train.shape[1]-1))
X_test = test[:,0:-1]
y_test = test[:,-1]
print(X_test.shape)
print(y_test.shape)

X_test = X_test.reshape(12,1,5)
#y_test = test[:,-1]
pred = lstm_model.predict(X_test, batch_size =1)
print(pred)
print("===================")
y_test = y_test.reshape(12,1)
print(y_test)
print(pred - y_test)

print("==========================================================================================")



predictions = list()
for i in range(len(test)):
    X2,y = test[i,0:-1], test[i,-1]
    #X2 = X2.reshape(12,1,5)
    yhat = forecast_lstm(lstm_model, 1, X2)
    predictions.append(yhat)
    expected = X[len(train)+i+1]
    print('Month = %d, Predicted = %f, Expected = %f' %(i+1, yhat, expected))



#repeat experiment
repeats = 10
n1 = len(test)
n2 = repeats
predictions1 = np.zeros((n1,n2), dtype = float)
for r in range(repeats):
    lstm_model = fit_lstm(train,1,10,100)
    # forecast the entire training dataset to build up state for forecasting
    train_reshaped = train[:,0:-1].reshape(19,1,5)
    lstm_model.predict(train_reshaped, batch_size =1)
    predictions2 = list()
    for i in range(len(test)):
        X, y = test[i,0:-1], test[i,-1]
        yhat = forecast_lstm(lstm_model, 1, X)
        predictions2.append(yhat)
    predictions1[:,r] = predictions2


print(predictions1)
print("============= mean ==================")
print(np.mean(predictions1, axis = 1))
print("=====================================")
#print(np.mean(predictions1, axis = 0))
print(X[12:])

'''

## MULTIVARIATE TIME SERIES


df = pd.read_csv('Pollution-Data.csv')
#df = pd.read_csv('~/Tensorflow/DAQ/NDX2.csv')
print(df.head())
print("===========================")
print(df.tail())
print('============================')
print(df.shape)

# load data
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')
dataset = pd.read_csv('Pollution-Data.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)
# manually specify column names
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
# mark all NA values with 0
dataset['pollution'].fillna(0, inplace=True)
# drop the first 24 hours
dataset = dataset[24:]
# summarize first 5 rows
print(dataset.head(5))
print(dataset.shape)
# save to file
dataset.to_csv('pollution.csv')


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# load dataset
dataset = pd.read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
# integer encode direction

encoder = LabelEncoder()
##values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
##scaler = MinMaxScaler(feature_range=(0, 1))
##scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(values, 1, 1) #(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head())


'''


# split into train and test sets
values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

'''

## MULTISTEP TIME SERIES PREDICTION

'''

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
from numpy import array
 
# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
 
# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
	# extract raw values
	raw_values = series.values
	# transform data to be stationary
	diff_series = difference(raw_values, 1)
	diff_values = diff_series.values
	diff_values = diff_values.reshape(len(diff_values), 1)
	# rescale values to -1, 1
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled_values = scaler.fit_transform(diff_values)
	scaled_values = scaled_values.reshape(len(scaled_values), 1)
	# transform into supervised learning problem X, y
	supervised = series_to_supervised(scaled_values, n_lag, n_seq)
	supervised_values = supervised.values
	# split into train and test sets
	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
	return scaler, train, test
 
# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
	# reshape training into [samples, timesteps, features]
	X, y = train[:, 0:n_lag], train[:, n_lag:]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	# design network
	model = Sequential()
	model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(y.shape[1]))
	model.compile(loss='mean_squared_error', optimizer='adam')
	# fit network
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
		model.reset_states()
	return model
 
# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, 1, len(X))
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]
 
# evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 0:n_lag], test[i, n_lag:]
		# make forecast
		forecast = forecast_lstm(model, X, n_batch)
		# store the forecast
		forecasts.append(forecast)
	return forecasts
 
# invert differenced forecast
def inverse_difference(last_ob, forecast):
	# invert first forecast
	inverted = list()
	inverted.append(forecast[0] + last_ob)
	# propagate difference forecast using inverted first value
	for i in range(1, len(forecast)):
		inverted.append(forecast[i] + inverted[i-1])
	return inverted
 
# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
	inverted = list()
	for i in range(len(forecasts)):
		# create array from forecast
		forecast = array(forecasts[i])
		forecast = forecast.reshape(1, len(forecast))
		# invert scaling
		inv_scale = scaler.inverse_transform(forecast)
		inv_scale = inv_scale[0, :]
		# invert differencing
		index = len(series) - n_test + i - 1
		last_ob = series.values[index]
		inv_diff = inverse_difference(last_ob, inv_scale)
		# store
		inverted.append(inv_diff)
	return inverted
 
# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = [row[i] for row in test]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))
 
# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):
	# plot the entire dataset in blue
	pyplot.plot(series.values)
	# plot the forecasts in red
	for i in range(len(forecasts)):
		off_s = len(series) - n_test + i - 1
		off_e = off_s + len(forecasts[i]) + 1
		xaxis = [x for x in range(off_s, off_e)]
		yaxis = [series.values[off_s]] + forecasts[i]
		pyplot.plot(xaxis, yaxis, color='red')
	# show the plot
	pyplot.show()
 
# load dataset
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# configure
n_lag = 1
n_seq = 3
n_test = 10
n_epochs = 1500
n_batch = 1
n_neurons = 1
# prepare data
scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
# fit model
model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
# make forecasts
forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)
# inverse transform forecasts and test
forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
actual = [row[n_lag:] for row in test]
actual = inverse_transform(series, actual, scaler, n_test+2)
# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_lag, n_seq)
# plot forecasts
plot_forecasts(series, forecasts, n_test+2)

'''
