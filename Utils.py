# -*- coding: utf-8 -*-


from google.colab import drive
drive.mount('/content/drive')


import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# Load and preprocess data
def load_data():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    test_dataset = h5py.File('test_catvnoncat.h5', "r")

    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # Training set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # Training set labels

    test_set_x_orig = np.array(test_dataset["test_set_x"][:])     # Test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])     # Test set labels

    classes = np.array(test_dataset["list_classes"][:])           # List of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# Load data
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Flatten and normalize
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T  # Shape: (12288, m_train)
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T     # Shape: (12288, m_test)

train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

n_x = train_x.shape[0]
n_y = 1

# Define activation functions and their derivatives
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)  # Convert dA to a correct object.
    dZ[Z <= 0] = 0
    return dZ

# Define initialization and forward/backward functions
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    
    W1 = np.random.randn(n_h, n_x) * np.sqrt(2. / n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * np.sqrt(2. / n_h)
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)
    return A, cache

def compute_cost(AL, Y):
    m = Y.shape[1]
    epsilon = 1e-15  # Small value to prevent log(0)
    cost = (-1 / m) * np.sum(Y * np.log(AL + epsilon) + (1 - Y) * np.log(1 - AL + epsilon))
    cost = np.squeeze(cost)
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # Number of layers
    
    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
    return parameters

def predict(X, Y, parameters):
    L = len(parameters) // 2
    A = X
    
    # Forward propagation
    for l in range(1, L):
        A_prev = A
        A, _ = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
    
    AL, _ = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
    predictions = (AL > 0.5).astype(int)
    
    accuracy = np.mean(predictions == Y) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    return predictions
