import numpy as np

'''
Written answer to Q4 is present in q4.md
'''
def softmax(x):
    return np.log(1 + np.exp(x))

def initialize_parameters(X, Y, hidden_size, variance):
    input_size = X.shape[0]
    output_size = Y.shape[0]
    std_dev1 = np.sqrt(variance / input_size)
    std_dev2 = np.sqrt(variance / hidden_size)
    W1 = np.random.normal(0, std_dev1, (hidden_size, input_size))
    W2 = np.random.normal(0, std_dev2, (output_size, hidden_size))
    return {'W1': W1, 'W2': W2}

def forward_propagation(X, theta):
    Z1 = np.dot(theta['W1'], X)
    A1 = np.tanh(Z1)
    Z2 = np.dot(theta['W2'], A1)
    y = softmax(Z2)
    return y, {'Z1': Z1, 'Z2': Z2, 'A1': A1, 'y': y}

def loss(predict, actual):
    m = actual.shape[1]
    loss_ = (1 / (2 * m)) * np.sum(np.square(predict - actual))
    return loss_

def back_propagation(X, Y, params, cache):
    m = X.shape[1]
    dy = cache['y'] - Y
    dW2 = (1 / m) * np.dot(dy, np.transpose(cache['A1']))
    dZ1 = np.dot(np.transpose(params['W2']), dy) * (1-np.power(cache['A1'], 2))
    dW1 = (1 / m) * np.dot(dZ1, np.transpose(X))
    return {"dW1": dW1, "dW2": dW2}

def update_parameters(gradient, theta, learning_rate = 0.01):
    W1 = theta['W1'] - learning_rate * gradient['dW1']
    W2 = theta['W2'] - learning_rate * gradient['dW2']
    return {'W1': W1, 'W2': W2}

def train(X, Y, learning_rate, hidden_size, variance, number_of_iterations = 50000):
    theta = initialize_parameters(X, Y, hidden_size, variance)
    cost_ = []
    for j in range(number_of_iterations):
        y, cache = forward_propagation(X, theta)
        cost_iteration = loss(y, Y)
        gradient = back_propagation(X, Y, theta, cache)
        theta = update_parameters(gradient, theta, learning_rate)
        cost_.append(cost_iteration)
    return theta, cost_, y

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
    for i in range(1, 101):
         X.append([i/10])
         Y.append([math.sin(X[i-1][0])])
    X = np.array(X)
    Y = np.array(Y)
    X *= 3.14
    Y += getNoise(Y.shape)
    return X, Y

initialization_variances = []
output_variances = []
for i in range(0, 999):
    variance = i / 10000
    initialization_variances.append(variance)
    X, Y = generateData()
    theta, loss_, y = train(X, Y, 0.0005, 50, variance, 10)
    output_variances.append(np.var(y[:50]))


import matplotlib.pyplot as plt
plt.plot(initialization_variances)
plt.plot(output_variances)
plt.show()