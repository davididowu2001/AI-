import numpy as np
import pandas as pd
#data prep

X = np.matrix([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.matrix([[0], [1], [1], [0]])

#formulas
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def mean_squared_error(y_pred, y_true):
    return ((y_pred - y_true)**2).sum() / (2*y_pred.size)

def accuracy(y_pred, y_true):
    acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)
    return acc.mean()


'''step 1: initialisation'''

#create variables needed

#number of hidden nodes
hidden_layer=4

#number of input nodes
input_size=5

#number of output
ouput_node=1

learningparameter=0.1
iterations=4000

'''step 1b:add weights to layers'''

#weight for hidden layer
hidden_layer_weights= np.random.randn(input_size,ouput_node)

#weight for output layer
output_weight=(hidden_layer,ouput_node)


'''step 2: forward pass'''

for data in range(iterations):
#forward pass
    #on hidden layer
    S1=np.dot(X,hidden_layer_weights)
    A1=sigmoid(S1)
    print(A1)

    #on output layer
    S2=np.dot(output_weight,A1)
    A2=sigmoid(S2)
    print(A1)

#backward pass
    #outer layer
    F1= S2 - Y
    delta1= F1 *A2*(1-A2)

    #hidden
    F2=np.dot(delta1,hidden_layer_weights.T)
    delta2=F2*A1*(1-A1)

#update weights
    hidden_layer_weights_update= np.dot(A1.T,delta1)
    output_weight_update=np.dot(X.T,delta2)

    hidden_layer_weights=hidden_layer_weights-learningparameter*hidden_layer_weights_update
    output_weight=output_weight_update-learningparameter*hidden_layer_weights_update

#forwardpass
    #hidden layer
    S1=np.dot(X,hidden_layer_weights)
    A1=sigmoid(S1)
    
    
    #outerlayer
    S2=np.dot(output_weight,A1)
    A2=sigmoid(S2)
    print(A2)

import numpy as np
#data prep

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

#formulas
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def mean_squared_error(y_pred, y_true):
    return ((y_pred - y_true)**2).sum() / (2*y_pred.size)

def accuracy(y_pred, y_true):
    acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)
    return acc.mean()


'''step 1: initialisation'''

#create variables needed

#number of hidden nodes
hidden_layer=4

#number of input nodes
input_size=5

#number of output
ouput_node=1

learningparameter=0.1
iterations=4000

'''step 1b:add weights to layers'''

#weight for hidden layer
hidden_layer_weights= np.random.randn(input_size,ouput_node)

#weight for output layer
output_weight=(hidden_layer,ouput_node)


'''step 2: forward pass'''

for data in range(iterations):
#forward pass
    #on hidden layer
    S1=np.dot(X,hidden_layer_weights)
    A1=sigmoid(S1)
    print(A1)

    #on output layer
    S2=np.dot(output_weight,A1)
    A2=sigmoid(S2)
    print(A1)

#backward pass
    #outer layer
    F1= S2 - Y
    delta1= F1 *A2*(1-A2)

    #hidden
    F2=np.dot(delta1,hidden_layer_weights.T)
    delta2=F2*A1*(1-A1)

#update weights
    hidden_layer_weights_update= np.dot(A1.T,delta1)
    output_weight_update=np.dot(X.T,delta2)

    hidden_layer_weights=hidden_layer_weights-learningparameter*hidden_layer_weights_update
    output_weight=output_weight_update-learningparameter*hidden_layer_weights_update

#forwardpass
    # #hidden layer
    # S1=np.dot(X,hidden_layer_weights)
    # A1=sigmoid(S1)
    #
    #
    # #outerlayer
    # S2=np.dot(output_weight,A1)
    # A2=sigmoid(S2)
    # print(A2)

