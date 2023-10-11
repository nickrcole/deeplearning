import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_loss(y_true, y_pred):
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def train_newton(X, y, theta):
    losses = []  # Store the loss values at each iteration
    while True:
        p = np.array(sigmoid(X.dot(theta[:, 0])), ndmin=2).T
        W = np.diag((p * (1 - p))[:, 0])
        hessian = X.T.dot(W).dot(X)
        grad = X.T.dot(y - p)
        step = np.dot(np.linalg.inv(hessian), grad)

        if np.linalg.norm(grad) < 0.01:
            break
        theta = theta + step

        # Calculate and store the loss value
        loss = logistic_loss(y, p)
        losses.append(loss[0])
    
    return losses

def train_gd(X, y, theta, learning_rate=0.001):
    losses = []  # Store the loss values at each iteration
    while True:
        p = np.array(sigmoid(X.dot(theta[:, 0])), ndmin=2).T
        grad = X.T.dot(y - p)
        if np.linalg.norm(grad) < 0.1:
            break
        step = learning_rate * grad
        theta = theta + step

        # Calculate and store the loss value
        loss = logistic_loss(y, p)
        losses.append(loss[0])
    
    return losses

def test_model(X, y, theta):
    prob = np.array(sigmoid(X.dot(theta)))
    
    prob = np.greater(prob, 0.5*np.ones((prob.shape[1],1)))
    accuracy = np.count_nonzero(np.equal(prob, y))/prob.shape[0] * 100

    return accuracy

def main():
    data = pd.read_csv("q4data.csv")

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X = np.c_[np.ones((X.shape[0], 1)), X]
    y = y[:, np.newaxis]

    theta = np.zeros((4,1))

    gd_losses = train_gd(X, y, theta)
    newton_losses = train_newton(X, y, theta)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(gd_losses, label="Gradient Descent Loss")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_title("Gradient Descent")
    ax1.legend()

    ax2.plot(newton_losses, label="Newton's Method Loss")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss")
    ax2.set_title("Newton's Method")
    ax2.legend()

    plt.tight_layout()
    plt.show()
    

main()