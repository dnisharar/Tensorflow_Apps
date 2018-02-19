
#!/usr/bin/python3
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
X = X[(len(X)-515):,1]
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

##test = np.ones((lag,lag),dtype = float)
##test[:future,:]= X_test
##print(test.shape)

test2_ = np.ones((forward+future, lag),dtype = float)
t1 = test2_.shape[0]
print(t1)
t2 = test2_.shape[1]
print(t2)
test2 = tf.placeholder(tf.float64, [t1, t2])
test2 = test2_

test2[0,:] = df[(n1-1),1:]
print(test2.shape)
print(test2[0,:])
print(test2[1,:])
#print(test2)

leng = forward #+ future 
y_predOLS =np.ones((leng,1),dtype=float) 
y_predSVR =np.ones((leng,1),dtype=float) 
y_predMLP =np.ones((leng,1),dtype=float) 
y_predMLP2 =np.ones((leng,1),dtype=float) 
y_predGLM =np.ones((leng,1),dtype=float) 

y_predRNN =np.ones((leng,1),dtype=float)
y_predRNN2 =np.ones((leng,1),dtype=float)
y_predLSTM =np.ones((leng,1),dtype=float)
y_predGRU =np.ones((leng,1),dtype=float)
y_predBIDIREC =np.ones((leng,1),dtype=float)
y_predWAVE =np.ones((leng,1),dtype=float)
y_predCNN =np.ones((leng,1),dtype=float)
### OLS 
print("+++++++++++++ OLS using scikit-learn +++++++++++++++++++++++++++++++++++=")
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

print("++++++++++++ SVR using scikit-learn +++++++++++++++++++++++++++++")
'''
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
mod = SVR(kernel='rbf', C= 0.02, gamma= 'auto', epsilon = 0.005).fit(X_train, y_train)
y_predSVR[0]     = mod.predict(test2[0,:].reshape(-1,lag)) 

for k in range(1,leng):
    y_predSVR[k]     = mod.predict(test2[k-1,:].reshape(-1,lag))
    test2[k,:(lag-1)] = test2[k-1,1:] 
    test2[k,(lag-1)] = y_predSVR[k] 
     

dif = y_test - y_predSVR
print(dif)

## GLM

'''
print("+++++++++++++++++ GLM using statsmodels ++++++++++++++++++++++++++++++++++++")

import scipy.stats as stats
import statsmodels.api as sm

from pyglmnet import GLM

y_t = np.asarray(y_train)
X_t = np.asarray(X_train)

glm_reg = sm.GLM(y_t, X_t, family=sm.families.Gamma())
glm_model = glm_reg.fit()
print(glm_model)

y_predGLM[0] = glm_model.predict(test2[0,:].reshape(-1,lag))

for k in range(1,leng):
    y_predGLM[k]     = glm_model.predict(test2[k-1,:].reshape(-1,lag))
    test2[k,:(lag-1)] = test2[k-1,1:] 
    test2[k,(lag-1)] = y_predGLM[k] 
 

print(y_test - y_predGLM)
'''

'''
print("+++++++++++++++++ regularized GLM using pyglmnet ++++++++++++++++++++++++++++++++++++")
import pyglmnet
from pyglmnet import GLM
from scipy import special
from scipy.special import expit

glm_model = GLM(distr = 'gamma', alpha = 0.01, reg_lambda = np.array([0.02, 0.08]), verbose = False)
#glm_model.threshold = 1e-5
glm_model.fit(X_train, y_train)

y_predGLM[0] = glm_model.predict(test2[0,:].reshape(-1,lag))

for k in range(1,leng):
    y_predGLM[k]     = glm_model.predict(test2[k-1,:].reshape(-1,lag))
    test2[k,:(lag-1)] = test2[k-1,1:] 
    test2[k,(lag-1)] = y_predGLM[k] 
 

print(y_test - y_predGLM)

'''


### NEURAL NETWORK 

print("+++++++++++++ MLP using scikit-learn +++++++++++++++++++")
MLP_model = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 50, 50), random_state=1)
MLP_model.fit(X_train, y_train)

y_predMLP[0] = MLP_model.predict(test2[0,:].reshape(-1,lag))

for k in range(1,leng):
    y_predMLP[k]     = MLP_model.predict(test2[k-1,:].reshape(-1,lag))
    test2[k,:(lag-1)] = test2[k-1,1:] 
    test2[k,(lag-1)] = y_predMLP[k] 
 

print(y_test - y_predMLP)



### TENSORFLOW 

print('++++++++++++++++++ MLP using Tensorflow first principles ++++++++++++++++++++++++++++++++')

in_dim = X_train.shape[0]
out_dim = X_train.shape[1]

# Create and train a tensorflow model of a neural network
def create_train_model(hidden_nodes, num_iters):

    # Reset the graph
    tf.reset_default_graph()

    # Placeholders for input and output data
    X = tf.placeholder(shape=(X_train.shape[0], X_train.shape[1]), dtype=tf.float64, name='X')
    y = tf.placeholder(shape=(X_train.shape[0], ), dtype=tf.float64, name='y')

    # Variables for two group of weights between the three layers of the network
    W1 = tf.Variable(np.random.rand(X_train.shape[1], hidden_nodes), dtype=tf.float64)
    #b1 = tf.Variable(tf.random_normal([hidden_nodes]), dtype=tf.float64)
    b1 = tf.Variable(np.random.rand(hidden_nodes), dtype=tf.float64)
    W2 = tf.Variable(np.random.rand(hidden_nodes, hidden_nodes), dtype=tf.float64)
    b2 = tf.Variable(np.random.rand(hidden_nodes), dtype=tf.float64)
    W3 = tf.Variable(np.random.rand(hidden_nodes, hidden_nodes), dtype=tf.float64)
    b3 = tf.Variable(np.random.rand(hidden_nodes), dtype=tf.float64)
    W4 = tf.Variable(np.random.rand(hidden_nodes, hidden_nodes), dtype=tf.float64)
    b4 = tf.Variable(np.random.rand(hidden_nodes), dtype=tf.float64)
    W5 = tf.Variable(np.random.rand(hidden_nodes, 1), dtype=tf.float64)
    b5 = tf.Variable(np.random.rand(1), dtype=tf.float64)

    # Create the neural net graph
    #A1 = tf.sigmoid( tf.add(tf.matmul(X, W1), b1))
    A1 = tf.nn.relu( tf.add(tf.matmul(X, W1), b1))
    #A2 = tf.sigmoid( tf.add(tf.matmul(A1, W2), b2))
    A2 = tf.nn.relu( tf.add(tf.matmul(A1, W2), b2))
    #A3 = tf.sigmoid( tf.add(tf.matmul(A2, W3), b3))
    A3 = tf.nn.relu( tf.add(tf.matmul(A2, W3), b3))
    #A4 = tf.sigmoid( tf.add(tf.matmul(A3, W4), b4))
    A4 = tf.nn.relu( tf.add(tf.matmul(A3, W4), b4))
    #y_est = tf.sigmoid( tf.add(tf.matmul(A4, W5), b5))
    y_est = tf.nn.relu( tf.add(tf.matmul(A4, W5), b5))

    # Define a loss function
    deltas = tf.square(y_est - y)
    #print("deltas " + deltas)
    loss = tf.reduce_sum(deltas)
    #print("loss " + loss)
    # Define a train operation to minimize the loss
    optimizer = tf.train.GradientDescentOptimizer(0.005)
    #print("optimizer " + optimizer)
    train = optimizer.minimize(loss)
    #print("train " + train)
    # Initialize variables and run session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Go through num_iters iterations
    for i in range(num_iters):
        sess.run(train, feed_dict={X: X_train, y: y_train})
        #loss_plot.append(sess.run(loss, feed_dict={X: X_train.as_matrix(), y: y_train.as_matrix()}))
        loss_plot.append(sess.run(loss, feed_dict={X: X_train, y: y_train}))
        weights1 = sess.run(W1)
        bias1    = sess.run(b1)
        weights2 = sess.run(W2)
        bias2    = sess.run(b2)
        weights3 = sess.run(W3)
        bias3    = sess.run(b3)
        weights4 = sess.run(W4)
        bias4    = sess.run(b4)
        weights5 = sess.run(W5)
        bias5    = sess.run(b5)
    
    print("loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, num_iters, loss_plot[-1]))
    #print(loss_plot)
    sess.close()
    return weights1, bias1, weights2, bias2, weights3, bias3, weights4, bias4, weights5, bias5



#loss_plot = []  
#create_train_model(100, 2000)
#create_train_model(100, 200)
#create_train_model(200, 300)


# Plot the loss function over iterations
#num_hidden_nodes = 10  
loss_plot = []  
weights1 = [] 
bias1 = [] 
weights2 = [] 
bias2 = []  
weights3 = [] 
bias3 = [] 
weights4 = [] 
bias4 = [] 
weights5 = [] 
bias5 = [] 
num_iters = 50

hidden_nodes = 10
weights1, bias1, weights2, bias2, weights3, bias3, weights4, bias4, weights5, bias5 = create_train_model(hidden_nodes, num_iters)
plt.plot(range(num_iters), loss_plot, label="nn: 4-%d-%d-%d-%d-3?" % (hidden_nodes,  hidden_nodes, hidden_nodes, hidden_nodes))

plt.xlabel('Iteration', fontsize=12)  
plt.ylabel('Loss', fontsize=12)  
plt.legend(fontsize=12)  
plt.show()

# Evaluate models on the test set
X = tf.placeholder(shape=(1, X_train.shape[1]), dtype=tf.float64, name='X')  
y = tf.placeholder(shape=(1, 1), dtype=tf.float64, name='y')


    # Forward propagation
W1 = tf.Variable(weights1)
b1 = tf.Variable(bias1)
W2 = tf.Variable(weights2)
b2 = tf.Variable(bias2)
W3 = tf.Variable(weights3)
b3 = tf.Variable(bias3)
W4 = tf.Variable(weights4)
b4 = tf.Variable(bias4)
W5 = tf.Variable(weights5)
b5 = tf.Variable(bias5)

#A1 = tf.sigmoid( tf.add(tf.matmul(X, W1), b1))
A1 = tf.nn.relu( tf.add(tf.matmul(X, W1), b1))
#A2 = tf.sigmoid( tf.add(tf.matmul(A1, W2), b2))
A2 = tf.nn.relu( tf.add(tf.matmul(A1, W2), b2))
#A3 = tf.sigmoid( tf.add(tf.matmul(A2, W3), b3))
A3 = tf.nn.relu( tf.add(tf.matmul(A2, W3), b3))
#A4 = tf.sigmoid( tf.add(tf.matmul(A3, W4), b4))
A4 = tf.nn.relu( tf.add(tf.matmul(A3, W4), b4))
#y_est = tf.sigmoid( tf.add(tf.matmul(A4, W5), b5))
y_est = tf.nn.relu( tf.add(tf.matmul(A4, W5), b5))
#print(y_est)


    # Calculate the predicted outputs
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    y_predMLP2[0] = sess.run(y_est, feed_dict={X: test2[0,:].reshape(-1,lag)}) 
    #y_predMLP[0] = MLP_model.predict(test2[0,:].reshape(-1,lag))

    for k in range(1,leng):
        #y_predMLP2[k]     = MLP_model.predict(test2[k-1,:].reshape(-1,lag))
        y_predMLP2[k]     = sess.run(y_est, feed_dict={X: test2[k-1,:].reshape(-1,lag)}) 
        test2[k,:(lag-1)] = test2[k-1,1:] 
        test2[k,(lag-1)] = y_predMLP2[k] 
 

#print(y_test - y_predMLP2)
print("MLP2")
print(y_predMLP2)

'''
    # Calculate the prediction accuracy
correct = [estimate.argmax(axis=0) == target.argmax(axis=0) 
            for estimate, target in zip(y_est_np, ytest.as_matrix())]
accuracy = 100 * sum(correct) / len(correct)
print(len(correct))
print(sum(correct))
print("accuracy = %.2f%%" %accuracy)
print(correct)
##print('Network architecture 4-%d-3, accuracy: %.2f%%' % (hidden_nodes, accuracy))
print("Network architecture = 4-%d-%d-%d-%d-3, accuracy = %.2f%%" % (hidden_nodes,  hidden_nodes, hidden_nodes, hidden_nodes,accuracy))
'''

'''
print('++++++++++++++++++ MLP using Tensorflow API ++++++++++++++++++++++++++++++++')

'''
### SIMPLE RNN


print("+++++++++++++++++++++++ Simple RNN using First Principles in Tensorflow ++++++++++++++++++++++++++")

num_epochs = 2
##total_series_length = 500
truncated_backprop_length = lag
state_size = 4
num_classes = 1
batch_size = 5
##num_batches = total_series_length//batch_size//truncated_backprop_length
num_batches = X_train.shape[0]/batch_size


future = 5
print(X_train.shape)
print(y_train.shape)
y_train = y_train.reshape(495,1)
print(y_train.shape)
print(y_test.shape)
print(X_test.shape)
print(test2.shape)

print(num_batches)
print(lag)

tf.reset_default_graph()

batchX_placeholder = tf.placeholder(tf.float64, [batch_size, truncated_backprop_length])
#batchY_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.float64, [batch_size, num_classes])

init_state = tf.placeholder(tf.float64, [batch_size, state_size])


W1 = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float64)
#b1 = tf.Variable(np.zeros((1,state_size)), dtype=tf.float64)
b1 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)

W2 = tf.Variable(np.random.rand(2*state_size, state_size), dtype=tf.float64)
#b2 = tf.Variable(np.zeros((1,state_size)), dtype=tf.float64)
b2 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)

W3 = tf.Variable(np.random.rand(2*state_size, state_size), dtype=tf.float64)
#b2 = tf.Variable(np.zeros((1,state_size)), dtype=tf.float64)
b3 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)

W = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float64)
#b = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float64)
b = tf.Variable(np.random.rand(1,num_classes), dtype=tf.float64)

# Unpack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1) # axis 1
#labels_series = tf.unstack(batchY_placeholder, axis=1) # axis 1
labels_series = batchY_placeholder

print("====================")
print(batchX_placeholder.shape)
print(batchY_placeholder.shape)

print(inputs_series)
print(labels_series)

print(init_state.shape)
print("===============")
print(inputs_series[0])

# Forward pass
current_state = init_state
states_series = []
#states_series = tf.placeholder(tf.float64, [batch_size, truncated_backprop_length*state_size])
#states_series = tf.placeholder(tf.float64, [batch_size, state_size])

for current_input in inputs_series:
    ##current_input = inputs_series[0]
    current_input = tf.reshape(current_input, [batch_size, 1])
    input_and_state_concatenated = tf.concat([current_input, current_state],1)  # Increasing number of columns

    #next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W1) + b1)  # Broadcasted addition
    next_state = tf.nn.relu(tf.matmul(input_and_state_concatenated, W1) + b1)  # Broadcasted addition
    states_series.append(next_state)
    #states = tf.concat([states_series, next_state],1)#.append(next_state)
    #states_series = states
    current_state = next_state
print("++++++++ First output ++++++++++")
#_output1 = states_series[-1]
print(states_series[-1])
print(len(states_series))
_output1 = states_series


inputs_series2 = _output1 #tf.unstack(_output1, axis=0)
current_state = init_state
states_series2 = []

for current_input in inputs_series2:
    ##current_input = inputs_series2[0]
    #current_input = tf.reshape(current_input, [batch_size, 1])
    input_and_state_concatenated = tf.concat([current_input, current_state],1)  # Increasing number of columns

    #next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W1) + b1)  # Broadcasted addition
    next_state = tf.nn.relu(tf.matmul(input_and_state_concatenated, W2) + b2)  # Broadcasted addition
    states_series2.append(next_state)
    current_state = next_state
print("++++++++ Second output ++++++++++")
#_output2 = states_series2[-1]
print(states_series2[-1])
print(len(states_series2))
_output2 = states_series2


inputs_series3 = _output2 #tf.unstack(_output1, axis=0)
current_state = init_state
states_series3 = []

for current_input in inputs_series3:
    ##current_input = inputs_series3[0]
    #current_input = tf.reshape(current_input, [batch_size, 1])
    input_and_state_concatenated = tf.concat([current_input, current_state],1)  # Increasing number of columns

    #next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W1) + b1)  # Broadcasted addition
    next_state = tf.nn.relu(tf.matmul(input_and_state_concatenated, W3) + b3)  # Broadcasted addition
    states_series3.append(next_state)
    current_state = next_state
print("++++++++ Third output ++++++++++")
#_output3 = states_series3[-1]
print(states_series3[-1])
print(len(states_series3))
_output3 = states_series3[-1]

#logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
logits_series = tf.matmul(_output3, W) + b  #Broadcasted addition
print(logits_series)


print("++++++++++++++++++++")
##predictions_series = [tf.nn.relu(logits) for logits in logits_series]
predictions_series = tf.nn.relu(logits_series)
print(predictions_series)
print(labels_series)

#deltas = [tf.square(logits - labels) for logits, labels in zip(logits_series,labels_series) ] 
deltas = tf.square(logits_series - labels_series)
print(deltas)

total_loss = tf.reduce_sum(deltas)

print(total_loss)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)
print(train_step)


def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        #one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        one_hot_output_series = np.array(predictions_series)[batch_series_idx]
        #single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])
        single_output_series = np.array([(1 if out < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)



x = X_train
y = y_train
_current_state = np.zeros((batch_size, state_size))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        
        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            #start_idx = batch_idx * truncated_backprop_length
            start_idx = batch_idx * batch_size
            #end_idx = start_idx + truncated_backprop_length
            end_idx = start_idx + batch_size

            #batchX = x[:,start_idx:end_idx]
            batchX = x[start_idx:end_idx,:]
            #batchY = y[:,start_idx:end_idx]
            batchY = y[start_idx:end_idx]
            #_pred = sess.run(predictions_series, feed_dict={ batchX_placeholder:batchX} )
            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={ batchX_placeholder:batchX, batchY_placeholder:batchY, init_state:_current_state })

            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)
    #y1_pred = sess.run(predictions_series, feed_dict = {x:test2[0,:].reshape(-1,lag)})
    #y_predRNN[0] = sess.run(predictions_series, feed_dict={X: test2[0,:].reshape(-1,lag)}) 
    ##y_predMLP[0] = MLP_model.predict(test2[0,:].reshape(-1,lag))

    #for k in range(1,leng):
        ##y_predMLP2[k]     = MLP_model.predict(test2[k-1,:].reshape(-1,lag))
        #y_predRNN[k]     = sess.run(predictions_series, feed_dict={X: test2[k-1,:].reshape(-1,lag)}) 
        #test2[k,:(lag-1)] = test2[k-1,1:] 
        #test2[k,(lag-1)] = y_predRNN[k] 
 

#print(y_test - y_predRNN)
#print("RNN")
#print(y_predRNN)

plt.ioff()
plt.show()
#print(y1_pred)
print(_predictions_series)
print(_predictions_series.shape)

'''
print("+++++++++++++++++++++++ Simple RNN using Tensorflow API ++++++++++++++++++++++++++")

'''



print("+++++++++++++++++++++++ GRU using First Principles in Tensorflow ++++++++++++++++++++++++++")

num_epochs = 10
truncated_backprop_length = lag
state_size = 10 #4
num_classes = 1
batch_size = 11 #5
num_batches = X_train.shape[0]/batch_size


future = 5
print(X_train.shape)
print(y_train.shape)
y_train = y_train.reshape(495,1)
print(y_train.shape)
print(y_test.shape)
print(X_test.shape)
print(test2.shape)

print(num_batches)
print(lag)

tf.reset_default_graph()

batchX_placeholder = tf.placeholder(tf.float64, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.float64, [batch_size, num_classes])

init_state = tf.placeholder(tf.float64, [batch_size, state_size])


Wr1 = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float64)
br1 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)
Wz1 = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float64)
bz1 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)
Wh1 = tf.Variable(np.random.rand(2*state_size+1, state_size), dtype=tf.float64)
bh1 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)

Wr2 = tf.Variable(np.random.rand(2*state_size, state_size), dtype=tf.float64)
br2 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)
Wz2 = tf.Variable(np.random.rand(2*state_size, state_size), dtype=tf.float64)
bz2 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)
Wh2 = tf.Variable(np.random.rand(2*state_size+state_size, state_size), dtype=tf.float64)
bh2 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)

Wr3 = tf.Variable(np.random.rand(2*state_size, state_size), dtype=tf.float64)
br3 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)
Wz3 = tf.Variable(np.random.rand(2*state_size, state_size), dtype=tf.float64)
bz3 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)
Wh3 = tf.Variable(np.random.rand(2*state_size+state_size, state_size), dtype=tf.float64)
bh3 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)


W = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float64)
b = tf.Variable(np.random.rand(1,num_classes), dtype=tf.float64)


# Unpack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1) 
#labels_series = tf.unstack(batchY_placeholder, axis=1) 
labels_series = batchY_placeholder

print("====================")
print(batchX_placeholder.shape)
print(batchY_placeholder.shape)

print(inputs_series)
print(labels_series)

print(init_state.shape)
print("===============")
print(inputs_series[0])

# Forward pass
current_state = init_state
states_series = []

print("first layer")
for current_input in inputs_series:
    #current_input = inputs_series[0]
    print(current_input.shape)
    current_input = tf.reshape(current_input, [batch_size, 1])
    print(current_input.shape)
    input_and_state_concatenated = tf.concat([current_input, current_state],1)  # Increasing number of columns
    print(input_and_state_concatenated.shape)
    r = tf.sigmoid(tf.matmul(input_and_state_concatenated, Wr1) + br1) 
    print(r.shape)
    z = tf.sigmoid(tf.matmul(input_and_state_concatenated, Wz1) + bz1) 
    print(z.shape)
    h_hat = tf.nn.relu(tf.matmul(tf.concat([input_and_state_concatenated, r],1), Wh1) + bh1) 
    print(h_hat.shape)
    h = tf.add( tf.matmul( (1-z), tf.transpose(current_state)) , tf.matmul(z, tf.transpose(h_hat)) )
    print(h.shape)
    print("end")

    next_state = h[:,:state_size]    
    states_series.append(next_state)
    current_state = next_state


print("++++++++ First output ++++++++++")
print(states_series[-1])
print(len(states_series))
_output1 = states_series


inputs_series2 = _output1 
current_state = init_state
states_series2 = []

print("second layer")
for current_input in inputs_series2:
    #current_input = inputs_series2[0]
    print(current_input.shape)
    #current_input = tf.reshape(current_input, [batch_size, 1])
    print(current_input.shape)
    input_and_state_concatenated = tf.concat([current_input, current_state],1)  # Increasing number of columns
    print(input_and_state_concatenated.shape)
    r = tf.sigmoid(tf.matmul(input_and_state_concatenated, Wr2) + br2) 
    print(r.shape)
    z = tf.sigmoid(tf.matmul(input_and_state_concatenated, Wz2) + bz2) 
    print(z.shape)
    h_hat = tf.nn.relu(tf.matmul(tf.concat([input_and_state_concatenated, r],1), Wh2) + bh2) 
    print(h_hat.shape)
    h = tf.add( tf.matmul( (1-z), tf.transpose(current_state)) , tf.matmul(z, tf.transpose(h_hat)) )
    print(h.shape)
    next_state = h[:,:state_size]    
    states_series2.append(next_state)
    current_state = next_state

print("++++++++ Second output ++++++++++")
print(states_series2[-1])
print(len(states_series2))
_output2 = states_series2


inputs_series3 = _output2 
current_state = init_state
states_series3 = []

for current_input in inputs_series3:
    #current_input = inputs_series3[0]
    print(current_input.shape)
    #current_input = tf.reshape(current_input, [batch_size, 1])
    print(current_input.shape)
    input_and_state_concatenated = tf.concat([current_input, current_state],1)  # Increasing number of columns
    print(input_and_state_concatenated.shape)
    r = tf.sigmoid(tf.matmul(input_and_state_concatenated, Wr3) + br3) 
    print(r.shape)
    z = tf.sigmoid(tf.matmul(input_and_state_concatenated, Wz3) + bz3) 
    print(z.shape)
    h_hat = tf.nn.relu(tf.matmul(tf.concat([input_and_state_concatenated, r],1), Wh3) + bh3) 
    print(h_hat.shape)
    h = tf.add( tf.matmul( (1-z), tf.transpose(current_state)) , tf.matmul(z, tf.transpose(h_hat)) )
    print(h.shape)
    next_state = h[:,:state_size]    
    states_series3.append(next_state)
    current_state = next_state

print("++++++++ Third output ++++++++++")
print(states_series3[-1])
print(len(states_series3))
_output3 = states_series3[-1]
print(_output3)


#logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
logits_series = tf.matmul(_output3, W) + b  #Broadcasted addition
print(logits_series)


print("++++++++++++++++++++")
##predictions_series = [tf.nn.relu(logits) for logits in logits_series]
predictions_series = tf.nn.relu(logits_series)
print(predictions_series)

#deltas = [tf.square(logits - labels) for logits, labels in zip(logits_series,labels_series) ] 
deltas = tf.square(logits_series - labels_series)
print(deltas)
total_loss = tf.reduce_sum(deltas)

print(total_loss)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)
print(train_step)


def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        #one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        one_hot_output_series = np.array(predictions_series)[batch_series_idx]
        #single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])
        single_output_series = np.array([(1 if out < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)



x = X_train
y = y_train
_current_state = np.zeros((batch_size, state_size))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        
        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            #start_idx = batch_idx * truncated_backprop_length
            start_idx = batch_idx * batch_size
            #end_idx = start_idx + truncated_backprop_length
            end_idx = start_idx + batch_size

            #batchX = x[:,start_idx:end_idx]
            batchX = x[start_idx:end_idx,:]
            #batchY = y[:,start_idx:end_idx]
            batchY = y[start_idx:end_idx]
            #_pred = sess.run(predictions_series, feed_dict={ batchX_placeholder:batchX} )
            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={ batchX_placeholder:batchX, batchY_placeholder:batchY, init_state:_current_state })

            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)
    #y1_pred = sess.run(predictions_series, feed_dict = {x:test2[0,:].reshape(-1,lag)})
    #y_predRNN[0] = sess.run(predictions_series, feed_dict={X: test2[0,:].reshape(-1,lag)}) 
    ##y_predMLP[0] = MLP_model.predict(test2[0,:].reshape(-1,lag))

    #for k in range(1,leng):
        ##y_predMLP2[k]     = MLP_model.predict(test2[k-1,:].reshape(-1,lag))
        #y_predRNN[k]     = sess.run(predictions_series, feed_dict={X: test2[k-1,:].reshape(-1,lag)}) 
        #test2[k,:(lag-1)] = test2[k-1,1:] 
        #test2[k,(lag-1)] = y_predRNN[k] 
 

#print(y_test - y_predRNN)
#print("RNN")
#print(y_predRNN)

plt.ioff()
plt.show()
#print(y1_pred)
print(_predictions_series)
print(_predictions_series.shape)


'''
print("+++++++++++++++++++++++ GRU using Tensorflow API ++++++++++++++++++++++++++")

'''


print("+++++++++++++++++++++++ LSTM using First Principles in Tensorflow ++++++++++++++++++++++++++")

num_epochs = 10
truncated_backprop_length = lag
state_size = 4
num_classes = 1
batch_size = 5
num_batches = X_train.shape[0]/batch_size


future = 5
print(X_train.shape)
print(y_train.shape)
y_train = y_train.reshape(495,1)
print(y_train.shape)
print(y_test.shape)
print(X_test.shape)
print(test2.shape)

print(num_batches)
print(lag)

tf.reset_default_graph()

batchX_placeholder = tf.placeholder(tf.float64, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.float64, [batch_size, num_classes])

init_state = tf.placeholder(tf.float64, [batch_size, state_size])


Wf1 = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float64)
bf1 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)
Wi1 = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float64)
bi1 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)
Wc1 = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float64)
bc1 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)
Wo1 = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float64)
bo1 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)

Wf2 = tf.Variable(np.random.rand(2*state_size, state_size), dtype=tf.float64)
bf2 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)
Wi2 = tf.Variable(np.random.rand(2*state_size, state_size), dtype=tf.float64)
bi2 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)
Wc2 = tf.Variable(np.random.rand(2*state_size, state_size), dtype=tf.float64)
bc2 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)
Wo2 = tf.Variable(np.random.rand(2*state_size, state_size), dtype=tf.float64)
bo2 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)

Wf3 = tf.Variable(np.random.rand(2*state_size, state_size), dtype=tf.float64)
bf3 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)
Wi3 = tf.Variable(np.random.rand(2*state_size, state_size), dtype=tf.float64)
bi3 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)
Wc3 = tf.Variable(np.random.rand(2*state_size, state_size), dtype=tf.float64)
bc3 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)
Wo3 = tf.Variable(np.random.rand(2*state_size, state_size), dtype=tf.float64)
bo3 = tf.Variable(np.random.rand(1,state_size), dtype=tf.float64)


W = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float64)
b = tf.Variable(np.random.rand(1,num_classes), dtype=tf.float64)


# Unpack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1) 
#labels_series = tf.unstack(batchY_placeholder, axis=1) 
labels_series = batchY_placeholder

print("====================")
print(batchX_placeholder.shape)
print(batchY_placeholder.shape)

print(inputs_series)
print(labels_series)

print(init_state.shape)
print("===============")
print(inputs_series[0])

# Forward pass
current_state = init_state
#cell_state =  init_state
cell_state = tf.Variable(np.random.rand(batch_size, state_size), dtype=tf.float64)
states_series = []
cell_series = []
cell_series.append(cell_state)

print("first layer")
for current_input in inputs_series:

    #current_input = inputs_series[0]
    print(current_input.shape)
    current_input = tf.reshape(current_input, [batch_size, 1])
    print(current_input.shape)
    input_and_state_concatenated = tf.concat([current_input, current_state],1)  # Increasing number of columns
    print(input_and_state_concatenated.shape)
    f = tf.sigmoid(tf.matmul(input_and_state_concatenated, Wf1) + bf1) 
    print(f.shape)
    i = tf.nn.relu(tf.matmul(input_and_state_concatenated, Wi1) + bi1) 
    print(i.shape)
    c_hat = cell_series[-1] 
    print(c_hat.shape) 
    c = tf.nn.relu(tf.matmul(input_and_state_concatenated, Wc1) + bc1)  
    print(c.shape) 
    b = tf.add( tf.matmul(f, tf.transpose(c_hat)) , tf.matmul(i, tf.transpose(c)))
    b = b[:,:-1]
    print(b.shape)
    cell_series.append(b)
    #print(cell_series)
    o = tf.nn.relu(tf.matmul(input_and_state_concatenated, Wo1) + bo1) 
    print(o.shape)
    h = tf.matmul( tf.nn.relu(b), tf.transpose(o))
    h = h[:,:-1]
    print(h.shape)
    states_series.append(h)
    #print(states_series)
    current_state = h
    print(current_state.shape)
    print("end")
    c_hat = b


print("++++++++ First output ++++++++++")
print(states_series[-1])
print(len(states_series))
_output1 = states_series


print("second layer")
inputs_series2 = _output1 
current_state = init_state
states_series2 = []
cell_state2 = tf.Variable(np.random.rand(batch_size, state_size), dtype=tf.float64)
cell_series2 = []
cell_series2.append(cell_state)


for current_input in inputs_series2:

    current_input = inputs_series2[0]
    print(current_input.shape)
    input_and_state_concatenated = tf.concat([current_input, current_state],1)  # Increasing number of columns
    print(input_and_state_concatenated.shape)
    f = tf.sigmoid(tf.matmul(input_and_state_concatenated, Wf2) + bf2) 
    print(f.shape)
    i = tf.nn.relu(tf.matmul(input_and_state_concatenated, Wi2) + bi2) 
    print(i.shape)
    c_hat = cell_series2[-1] 
    print(c_hat.shape) 
    c = tf.nn.relu(tf.matmul(input_and_state_concatenated, Wc2) + bc2)  
    print(c.shape) 
    b = tf.add( tf.matmul(f, tf.transpose(c_hat)) , tf.matmul(i, tf.transpose(c)))
    b = b[:,:-1]
    print(b.shape)
    cell_series2.append(b)
    #print(cell_series)
    o = tf.nn.relu(tf.matmul(input_and_state_concatenated, Wo2) + bo2) 
    print(o.shape)
    h = tf.matmul( tf.nn.relu(b), tf.transpose(o))
    h = h[:,:-1]
    print(h.shape)
    states_series2.append(h)
    #print(states_series)
    current_state = h
    print(current_state.shape)
    print("end")
    c_hat = b


print("++++++++ Second output ++++++++++")
print(states_series2[-1])
print(len(states_series2))
_output2 = states_series2


print("Third layer")
inputs_series3 = _output2 
current_state = init_state
states_series3 = []
cell_state3 = tf.Variable(np.random.rand(batch_size, state_size), dtype=tf.float64)
cell_series3 = []
cell_series3.append(cell_state)


for current_input in inputs_series3:

    current_input = inputs_series3[0]
    print(current_input.shape)
    input_and_state_concatenated = tf.concat([current_input, current_state],1)  # Increasing number of columns
    print(input_and_state_concatenated.shape)
    f = tf.sigmoid(tf.matmul(input_and_state_concatenated, Wf3) + bf3) 
    print(f.shape)
    i = tf.nn.relu(tf.matmul(input_and_state_concatenated, Wi3) + bi3) 
    print(i.shape)
    c_hat = cell_series3[-1] 
    print(c_hat.shape) 
    c = tf.nn.relu(tf.matmul(input_and_state_concatenated, Wc3) + bc3)  
    print(c.shape) 
    b = tf.add( tf.matmul(f, tf.transpose(c_hat)) , tf.matmul(i, tf.transpose(c)))
    b = b[:,:-1]
    print(b.shape)
    cell_series3.append(b)
    #print(cell_series)
    o = tf.nn.relu(tf.matmul(input_and_state_concatenated, Wo3) + bo3) 
    print(o.shape)
    h = tf.matmul( tf.nn.relu(b), tf.transpose(o))
    h = h[:,:-1]
    print(h.shape)
    states_series3.append(h)
    #print(states_series3)
    current_state = h
    print(current_state.shape)
    print("end")
    c_hat = b


print("++++++++ Third output ++++++++++")
print(states_series3[-1])
print(len(states_series3))
_output3 = states_series3[-1]
print(_output3)


#logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
logits_series = tf.matmul(_output3, W) + b  #Broadcasted addition
print(logits_series)

print("++++++++++++++++++++")
##predictions_series = [tf.nn.relu(logits) for logits in logits_series]
predictions_series = tf.nn.relu(logits_series)
print(predictions_series)

#deltas = [tf.square(logits - labels) for logits, labels in zip(logits_series,labels_series) ] 
deltas = tf.square(logits_series - labels_series)
print(deltas)
total_loss = tf.reduce_sum(deltas)

print(total_loss)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)
print(train_step)


def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        #one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        one_hot_output_series = np.array(predictions_series)[batch_series_idx]
        #single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])
        single_output_series = np.array([(1 if out < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)

'''
x = X_train
y = y_train
_current_state = np.zeros((batch_size, state_size))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        
        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            #start_idx = batch_idx * truncated_backprop_length
            start_idx = batch_idx * batch_size
            #end_idx = start_idx + truncated_backprop_length
            end_idx = start_idx + batch_size

            #batchX = x[:,start_idx:end_idx]
            batchX = x[start_idx:end_idx,:]
            #batchY = y[:,start_idx:end_idx]
            batchY = y[start_idx:end_idx]
            #_pred = sess.run(predictions_series, feed_dict={ batchX_placeholder:batchX} )
            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={ batchX_placeholder:batchX, batchY_placeholder:batchY, init_state:_current_state })

            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)
    #y1_pred = sess.run(predictions_series, feed_dict = {x:test2[0,:].reshape(-1,lag)})
    #y_predRNN[0] = sess.run(predictions_series, feed_dict={X: test2[0,:].reshape(-1,lag)}) 
    ##y_predMLP[0] = MLP_model.predict(test2[0,:].reshape(-1,lag))

    #for k in range(1,leng):
        ##y_predMLP2[k]     = MLP_model.predict(test2[k-1,:].reshape(-1,lag))
        #y_predRNN[k]     = sess.run(predictions_series, feed_dict={X: test2[k-1,:].reshape(-1,lag)}) 
        #test2[k,:(lag-1)] = test2[k-1,1:] 
        #test2[k,(lag-1)] = y_predRNN[k] 
 

#print(y_test - y_predRNN)
#print("RNN")
#print(y_predRNN)

plt.ioff()
plt.show()
#print(y1_pred)
print(_predictions_series)
print(_predictions_series.shape)
'''


'''
print("+++++++++++++++++++++++ LSTM using Tensorflow API ++++++++++++++++++++++++++")

'''



print("+++++++++++++++++++++++ CNN using First Principles in Tensorflow ++++++++++++++++++++++++++")

# Train the model using the training sets
'''
OLS_model.fit(X_train,y_train) 
y_predOLS[0] = OLS_model.predict(test2[0,:].reshape(-1,lag))
print(y_predOLS)

for k in range(1,leng):
    y_predOLS[k]     = OLS_model.predict(test2[k-1,:].reshape(-1,lag))
    test2[k,:(lag-1)] = test2[k-1,1:] 
    test2[k,(lag-1)] = y_predOLS[k] 

y_test = y_test.reshape(leng,1) 
print(y_test - y_predOLS)
'''

train_dataset = X_train 
train_labels  = y_train
test_dataset  = test2
test_labels   = y_predCNN

batch_size  = 10
image_width = train_dataset.shape[1] 
image_height = 1
image_depth = 1
 

from tflearn.layers.conv import conv_1d, max_pool_1d, avg_pool_1d
from tflearn.layers.core import fully_connected, dropout, flatten



def model_CNN(data):
    layer1_conv = conv_1d(data, nb_filter = 6, filter_size = 5, activation='relu', padding='SAME', bias = True)
    layer1_pool = avg_pool_1d(layer1_conv, kernel_size = 2, strides=2, padding='SAME')

    layer2_conv = conv_1d(layer1_pool, nb_filter = 16, filter_size = 5, strides = [1,1,1,1], activation='relu', padding='VALID', bias = True)
    layer2_pool = avg_pool_1d(layer2_conv, kernel_size = 2, strides=2, padding='SAME')

    flat_layer = flatten(layer2_pool)
    layer3_fccd = fully_connected(flat_layer, n_units = 120, activation='relu', bias = True)

    layer4_fccd = fully_connected(layer3_fccd, n_units = 84, activation='relu', bias = True)

    w = tf.Variable(tf.truncated_normal([84, 1], stddev=0.1))
    b = tf.Variable(tf.constant(1.0, shape = [1]))

    logits = tf.matmul(layer4_fccd, w) + b
    return logits

#number of iterations and learning rate
num_steps = 20
display_step = 5
learning_rate = 0.5 #0.001
num_labels = 1

graph = tf.Graph()
with graph.as_default():
    #1) First we put the input data in a tensorflow friendly form. 
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, image_depth))
    tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset, tf.float32)
 
    #2) Then, the weight matrices and bias vectors are initialized
 
    #3. The model used to calculate the logits (predicted labels)
    model = model_CNN
    logits = model(tf_train_dataset)

    #4. then we compute the softmax cross entropy between the logits and the (actual) labels
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    
    #5. The optimizer is used to calculate the gradients of the loss function 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
 
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_dataset))


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print(' Initialized with learning_rate', learning_rate)
    for step in range(num_steps):
 
        #Since we are using stochastic gradient descent, we are selecting  small batches from the training dataset,
        #and training the convolutional neural network each time with a batch. 
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        
        if step % display_step == 0:
            train_accuracy = accuracy(predictions, batch_labels)
            test_accuracy = accuracy(test_prediction.eval(), test_labels)
            message = "step {:04d} : loss is {:06.2f}, accuracy on training set {:02.2f} %, accuracy on test set {:02.2f} %".format(step, l, train_accuracy, test_accuracy)
            print(message)



'''
print("+++++++++++++++++++++++ CNN using Tensorflow API ++++++++++++++++++++++++++")

'''








