import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

'''
    For the life of me I couldn't get this to converge
    So it doesn't work but the attempt is there
'''
def train(X, y, theta, learning_rate=0.000001, tolerance=0.01, max_iter=1000):
    for i in range(max_iter):
        random_index = random.randint(0, X.shape[0]-1)
        x = np.array([X[random_index]])
        this_y = y[random_index]

        p = np.array(sigmoid(x.dot(theta)), ndmin=2).T
        grad = x.T.dot(this_y - p)
        
        if np.linalg.norm(grad) < tolerance:
            return theta
        
        step = learning_rate * grad
        theta = theta + step
    
    return theta

def main():
    data = pd.read_csv("q5data.csv")

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X = np.c_[np.ones((X.shape[0], 1)), X]
    y = y[:, np.newaxis]
    theta = np.zeros((3,1))

    theta = train(X, y, theta)
    print(theta)
    
main()