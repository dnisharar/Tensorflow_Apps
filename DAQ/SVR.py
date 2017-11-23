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


d1 = pd.read_csv('NDX2.csv')
forward = 5
future = 5
lag = 15
batch_size =  15

X = np.array(d1)
X = X[len(X)-515:,1]
print(X.shape)


def to_supervised(data, lag):
    df = pd.DataFrame(data)
    columns = [df.shift(lag + 1 -i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis =1)
    df.dropna(inplace =  True)
    return df

df = to_supervised(X, lag)
df = np.array(df)
print(df.shape)
n1 = len(df) - forward
print(n1)

train, test = df[:n1,:], df[n1:,:]
X_train, y_train = train[:,:-1], train[:,-1]
print(X_train.shape)
print(y_train.shape)
X_test, y_test = test[:,:-1], test[:,-1]
print(X_test.shape)
print(y_test.shape)

test = np.ones((lag,lag),dtype = float)
test[:future,:]= X_test
print(test.shape)

test2 = np.ones((forward+future, lag),dtype = float)
test2[0,:] = df[(n1-1),1:]
print(test2.shape)
print(test2[0,:])
print(test2[1,:])
#print(test2)

leng = forward #+ future 
y_predOLS =np.ones((leng,1),dtype=float) 
y_predSVR =np.ones((leng,1),dtype=float) 
y_predMLP =np.ones((leng,1),dtype=float) 

### OLS 

OLS_model = linear_model.LinearRegression()

# Train the model using the training sets
OLS_model.fit(X_train,y_train) 
y_predOLS[0] = OLS_model.predict(test2[0,:].reshape(-1,lag))
print(y_predOLS)

for k in range(1,leng):
    y_predOLS[k]     = OLS_model.predict(test2[k-1,:].reshape(-1,lag))
    test2[k,:(lag-1)] = test2[k-1,1:] 
    test2[k,(lag-1)] = y_predOLS[k] 

y_test = y_test.reshape(leng,1) 
print(y_test - y_predOLS)

###SVR 
'''
print("++++++++++++ SVR using scikit-learn +++++++++++++++++++++++++++++)
#Model Optimization 
#parameters    = {'kernel':('linear','poly', 'rbf'),'C':[1, 10, 100, 1000], 'gamma': np.logspace(-2, 1, 4,base=2),'epsilon':np.logspace(-2,1,4,base=10)} 
parameters    = {'kernel':('linear','poly', 'rbf'),'C':[1, 10], 'gamma': np.logspace(-2, 1, 2,base=2),'epsilon':np.logspace(-2,1,2,base=10)}
SVR_model     = SVR()
grid           = GridSearchCV(SVR_model, parameters)
SVR_model = grid.fit(X_train, y_train)
#SVR_model = SVR_model.fit(X_train, y_train)
y_predSVR[0]     = SVR_model.predict(test2[0,:].reshape(-1,lag)) 

#SVR_model_train = grid.fit(X_train, y_train) 
 
#SVR_model_optimized     = SVR(kernel = grid.best_params_["kernel"], C=grid.best_params_["C"], gamma=grid.best_params_["gamma"],epsilon = grid.best_params_["epsilon"]) 
#SVR_model_optimized.fit(X_train, y_train)
#y_predSVR[0]     = SVR_model_optimized.predict(test2[0,:].reshape(-1,lag)) 

 
for k in range(1,leng):
    y_predSVR[k]     = SVR_model.predict(test2[k-1,:].reshape(-1,lag))
    test2[k,:(lag-1)] = test2[k-1,1:] 
    test2[k,(lag-1)] = y_predSVR[k] 
     

dif = y_test - y_predSVR
print(dif)

'''


''' NEURAL NETWORK '''
print("+++++++++++++ MLP using scikit-learn +++++++++++++++++++")
MLP_model = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(400, 400, 400), random_state=1)
MLP_model.fit(X_train, y_train)

y_predMLP[0] = MLP_model.predict(test2[0,:].reshape(-1,lag))

for k in range(1,leng):
    y_predMLP[k]     = MLP_model.predict(test2[k-1,:].reshape(-1,lag))
    test2[k,:(lag-1)] = test2[k-1,1:] 
    test2[k,(lag-1)] = y_predMLP[k] 
 

print(y_test - y_predMLP)


'''
t = np.linspace(1,leng,leng)
print(t)
plt.plot(t,y_test)
plt.show()

lines = plt.plot(t, y_test, t, y_predMLP)
plt.show()

'''

### TENSORFLOW 

print('++++++++++++++++++ MLP using Tensorflow ++++++++++++++++++++++++++++++++')

# Create and train a tensorflow model of a neural network
def create_train_model(hidden_nodes, num_iters):

    # Reset the graph
    tf.reset_default_graph()

    # Placeholders for input and output data
    X = tf.placeholder(shape=(495, 15), dtype=tf.float64, name='X')
    y = tf.placeholder(shape=(495, ), dtype=tf.float64, name='y')

    # Variables for two group of weights between the three layers of the network
    W1 = tf.Variable(np.random.rand(15, hidden_nodes), dtype=tf.float64)
    W2 = tf.Variable(np.random.rand(hidden_nodes, 1), dtype=tf.float64)

    # Create the neural net graph
    A1 = tf.sigmoid(tf.matmul(X, W1))
    y_est = tf.sigmoid(tf.matmul(A1, W2))

    # Define a loss function
    deltas = tf.square(y_est - y)
    loss = tf.reduce_sum(deltas)

    # Define a train operation to minimize the loss
    optimizer = tf.train.GradientDescentOptimizer(0.005)
    train = optimizer.minimize(loss)

    # Initialize variables and run session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Go through num_iters iterations
    for i in range(num_iters):
        sess.run(train, feed_dict={X: X_train, y: y_train})
        #loss_plot[hidden_nodes].append(sess.run(loss, feed_dict={X: X_train.as_matrix(), y: y_train.as_matrix()}))
        loss_plot[hidden_nodes].append(sess.run(loss, feed_dict={X: X_train, y: y_train}))
        weights1 = sess.run(W1)
        weights2 = sess.run(W2)

    print("loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, num_iters, loss_plot[hidden_nodes][-1]))
    sess.close()
    return weights1, weights2


# Plot the loss function over iterations
hidden_nodes = 50 #[5, 10, 20]  
loss_plot = {50: []} #, 10: [], 20: []}  
weights1 = {50: None} #, 10: None, 20: None}  
weights2 = {50: None} #, 10: None, 20: None}  
num_iters = 2000


#plt.figure(figsize=(12,8))  
#for hidden_nodes in num_hidden_nodes:  
weights1[hidden_nodes], weights2[hidden_nodes] = create_train_model(hidden_nodes, num_iters)
plt.plot(range(num_iters), loss_plot[hidden_nodes], label="nn: in-%d-out" % hidden_nodes)

plt.xlabel('Iteration', fontsize=12)  
plt.ylabel('Loss', fontsize=12)  
plt.legend(fontsize=12)  
plt.show()


# Evaluate models on the test set
X = X_test
y = y_test

#for hidden_nodes in num_hidden_nodes:
    # Forward propagation
W1 = tf.Variable(weights1[hidden_nodes])
W2 = tf.Variable(weights2[hidden_nodes])
A1 = tf.sigmoid(tf.matmul(X, W1))
#A1 = tf.nn.relu(tf.matmul(X, W1))
y_est = tf.sigmoid(tf.matmul(A1, W2))
#y_est = tf.nn.relu(tf.matmul(A1, W2))
print(y_est)

'''
    # Calculate the predicted outputs
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    y_est_np = sess.run(y_est, feed_dict={X: X_test, y: y_test})

    # Calculate the prediction accuracy
correct = [estimate.argmax(axis=0) == target.argmax(axis=0) 
            for estimate, target in zip(y_est_np, ytest.as_matrix())]
accuracy = 100 * sum(correct) / len(correct)
print('Network architecture 4-%d-3, accuracy: %.2f%%' % (hidden_nodes, accuracy))

'''