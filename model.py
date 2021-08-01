import numpy as np
import matplotlib.pyplot as plt
import h5py
import copy
import activations

global layer_dimensions
global layer_activation_list

def deep_network_initialize_parameters(layer_dimensions):

    np.random.seed(1)
    num_layers = len(layer_dimensions)

    #initalize parameter dictionary
    parameter_dictionary = {}

    for l in range(1,num_layers):

        parameter_dictionary["W" + str(l)] = np.random.randn(layer_dimensions[l],layer_dimensions[l-1])/np.sqrt(layer_dimensions[l-1])
        parameter_dictionary["b" + str(l)] = np.zeros((layer_dimensions[l], 1))

    return parameter_dictionary

def forward_activation_calculation(A_previous, W, b, activation):

    Z = np.dot(W,A_previous) + b
    linear_cache = (A_previous, W, b)

    if activation == "sigmoid":
        A = activations.sigmoid(Z)
    elif activation == "relu":
        A = activations.relu(Z)
    elif activation == "tanh":
        A = activations.tanh(Z)
    elif activation == "softmax":
        A = activations.softmax(Z)

    linear_activation_cache = (linear_cache, Z)
    return A, linear_activation_cache

def forward_model(X, parameter_dictionary, layer_activation_list):

    #params list has Wi, bi for each layer, so /= 2
    L = len(parameter_dictionary)//2

    A = X
    cache_list = []
    for l in range(0, L):
        A_previous = A
        A, linear_activation_cache = forward_activation_calculation(A_previous, parameter_dictionary["W"+str(l+1)], parameter_dictionary["b"+str(l+1)], layer_activation_list[l])
        cache_list.append(linear_activation_cache)

    y_hat = A
    return y_hat, cache_list


def calculate_cost(y_hat, Y):

    m = Y.shape[1]
    cost = np.squeeze(-np.sum(Y*np.log(y_hat)+(1-Y)*np.log(1-y_hat), axis = 1, keepdims = True)/m)

    return cost

def cross_entropy_loss(y_hat, Y):
    m = Y.shape[1]
    cost = -np.mean(Y*np.log(y_hat+1e-6))
    return cost

def backward_activation_calculation(y_hat, Y, dA, current_cache, activation):

    linear_cache, Z = current_cache
    if activation == "sigmoid":
        dZ = activations.sigmoid_backprop(dA, Z)
    elif activation == "relu":
        dZ = activations.relu_backprop(dA, Z)
    elif activation == "tanh":
        dZ = activations.tanh_backprop(dA, Z)
    elif activation == "softmax":
        dZ = activations.softmax_backprop(y_hat, Y)

    #fetch from dictionary
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    #calculate gradients
    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis = 1, keepdims = True)/m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def backward_model(y_hat, Y, cache_list, layer_activation_list):


    L = len(cache_list)
    grads = {}

    dA_tmp = y_hat - Y

    for l in reversed(range(L)):
        current_cache = cache_list[l]
        dA, dW, db = backward_activation_calculation(y_hat, Y, dA_tmp, current_cache, activation = layer_activation_list[l])
        grads["dA"+str(l)], grads["dW"+str(l+1)], grads["db"+str(l+1)] = dA, dW, db
        dA_tmp = dA

    return grads

def update_params(parameter_dictionary, grads, learning_rate):

    learned_parameters = {}
    L = len(parameter_dictionary)//2

    for l in range(1,L+1):

        learned_parameters["W" + str(l)] = parameter_dictionary["W" + str(l)] - learning_rate*grads["dW"+str(l)]
        learned_parameters["b" + str(l)] = parameter_dictionary["b" + str(l)] - learning_rate*grads["db"+str(l)]

    return learned_parameters

def generate_model(X_train, Y_train, layer_dims, layer_activ_list, learning_rate, iterations):
    global layer_activation_list
    layer_activation_list = layer_activ_list
    global layer_dimensions
    layer_dimensions = layer_dims

    np.random.seed(1)
    assert(len(layer_activation_list) + 1 == len(layer_dimensions))
    parameter_dictionary = deep_network_initialize_parameters(layer_dimensions)
    costs = []
    print("Training Model ...")
    for i in range(iterations):

        y_hat, cache_list = forward_model(X_train, parameter_dictionary, layer_activation_list)
        cost = cross_entropy_loss(y_hat, Y_train)
        grads = backward_model(y_hat, Y_train, cache_list, layer_activation_list)
        parameter_dictionary = update_params(parameter_dictionary, grads, learning_rate)
        if i % 10 == 0 or i == iterations-1:
            costs.append(cost)
            # print("Cost for iteration "+str(i)+ ": " + str(cost))

    return parameter_dictionary, costs

def predict(X_test, Y_test, learned_parameters):

    m = X_test.shape[1]
    probabilities, cache = forward_model(X_test, learned_parameters, layer_activation_list)
    y_hat = np.argmax(probabilities, axis = 0)
    Y = np.argmax(Y_test, axis = 0)
    accuracy = (y_hat == Y).mean()
    print("Test set accuracy: " + str(accuracy))
    return accuracy

def model_predict(vectorized_input, learned_parameters):
    probabilities, cache = forward_model(vectorized_input, learned_parameters, layer_activation_list)
    y_hat = np.argmax(probabilities, axis = 0)
    return y_hat

def plot_cost(costs):
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.show()
