import numpy as np

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def initialize_parameters(X, Y, hidden_size):
    input_size = X.shape[0]
    output_size = Y.shape[0]
    W1 = np.random.randn(hidden_size, input_size)*np.sqrt(1/input_size)
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size)*np.sqrt(1/hidden_size)
    b2 = np.zeros((output_size, 1))
    return {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}

def forward_propagation(X, theta):
    Z1 = np.dot(theta['W1'], X)+theta['b1']
    A1 = np.tanh(Z1)
    Z2 = np.dot(theta['W2'], A1)+theta['b2']
    y = sigmoid(Z2)  
    return y, {'Z1': Z1, 'Z2': Z2, 'A1': A1, 'y': y}

def loss(predict, actual):
    m = actual.shape[1]
    loss_ = (1 / (2 * m)) * np.sum(np.square(predict - actual))
    return loss_

def back_propagation(X, Y, params, cache):
    m = X.shape[1]
    dy = cache['y'] - Y
    dW2 = (1 / m) * np.dot(dy, np.transpose(cache['A1']))
    db2 = (1 / m) * np.sum(dy, axis=1, keepdims=True)
    dZ1 = np.dot(np.transpose(params['W2']), dy) * (1-np.power(cache['A1'], 2))
    dW1 = (1 / m) * np.dot(dZ1, np.transpose(X))
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

def update_parameters(gradient, theta, learning_rate = 0.01):
    W1 = theta['W1'] - learning_rate * gradient['dW1']
    b1 = theta['b1'] - learning_rate * gradient['db1']
    W2 = theta['W2'] - learning_rate * gradient['dW2']
    b2 = theta['b2'] - learning_rate * gradient['db2']
    return {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}

def train(X, Y, learning_rate, hidden_size, number_of_iterations = 50000):
    theta = initialize_parameters(X, Y, hidden_size)
    cost_ = []
    for j in range(number_of_iterations):
        y, cache = forward_propagation(X, theta)
        cost_iteration = loss(y, Y)
        gradient = back_propagation(X, Y, theta, cache)
        theta = update_parameters(gradient, theta, learning_rate)
        cost_.append(cost_iteration)
    return theta, cost_

import math
def getNoise(size):
    variance = 0.1
    mean = 0
    std_dev = np.sqrt(variance)
    num_samples = size
    return np.random.normal(mean, std_dev, num_samples)

def generateData():
    X = []
    Y = []
    for i in range(1, 21):
         X.append([i/10])
         Y.append([math.sin(X[i-1][0])])
    X = np.array(X)
    Y = np.array(Y)
    X *= 3.14
    Y += getNoise(Y.shape)
    return X, Y

X, Y = generateData()
theta, loss_ = train(X, Y, 0.0005, 6, 5000)

import matplotlib.pyplot as plt
plt.plot(loss_)
plt.show()