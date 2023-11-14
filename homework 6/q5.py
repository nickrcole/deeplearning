import numpy as np
import math
import matplotlib.pyplot as plt


def fit_regularization(X, y, d, l=0.5): 
    a = [] 
    c = [] 
    for i in range(d): 
        row = [] 
        for j in range(d): 
            l = 0 if i != j else l 
            row.append(np.sum([x**(i+j) + l for x in X]))
        a.append(row) 
        c.append(np.sum(np.multiply([x**i for x in X], y))) 
    w = np.linalg.inv(a) * np.transpose(c) 
    return w

def fit(X, y, d):
    a = []
    c = []
    for i in range(d):
        row = []
        for j in range(d):
            row.append(np.sum([x**(i+j) for x in X]))
        a.append(row)
        c.append(np.sum(np.multiply([x**i for x in X], y)))
    w = np.linalg.inv(a) * np.transpose(c)
    return w

def predict(w, X): 
    pred = [] 
    for x in X:
        pred.append(np.sum([w_j * x**j for j, w_j in enumerate(w)])) 
    return pred

def getNoise(size):
    variance = 0.1
    mean = 0
    std_dev = np.sqrt(variance)
    num_samples = size
    return np.random.normal(mean, std_dev, num_samples)

def generateData():
    np.random.seed(0)
    X = np.arange(0, 10, 0.1)
    X = X + np.random.normal(0, 0.2, X.shape)
    y = np.sin(X)
    y = y + np.random.normal(0, 0.3, y.shape)
    return X, y

X, Y = generateData()
d = 10

plt.scatter(X, Y)
w = fit(X, Y, d)
plt.plot(X, predict(w, X), c="red")
w = fit_regularization(X, Y, d)
plt.plot(X, predict(w, X), c="green")
plt.plot(X, [np.sin(x_i) for x_i in X], c="orange")
plt.xlim(np.pi, 2*np.pi)
plt.show()