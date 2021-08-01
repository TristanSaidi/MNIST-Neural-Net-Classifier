import numpy as np

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A

def relu(Z):
    A = np.maximum(0,Z)
    return A

def tanh(Z):
    A = (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
    assert(A.shape == Z.shape)
    return A

def softmax(Z):
    exps = np.exp(Z - np.max(Z))
    return exps/np.sum(exps, axis = 0, keepdims = True)

def sigmoid_backprop(dA, Z):
    s = sigmoid(-Z)
    dZ = dA*s*(1-s)
    return dZ

def relu_backprop(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z < 0] = 0
    return dZ

def tanh_backprop(dA, Z):
    dZ = dA*(1-np.square(tanh(Z)))
    return dZ

def softmax_backprop(y_hat, y_train):
    dZ = y_hat - y_train
    return dZ
