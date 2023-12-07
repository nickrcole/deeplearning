import numpy as np
import csv
import matplotlib.pyplot as plt
import sys
import math

K_VALUE = 4
CONVERGENCE_THRESHOLD = 1e-4

def import_data():
    x = []
    y = []
    with open("./p5data.csv", "r") as data:
        reader = csv.reader(data)
        for row in reader:
            x.append(float(row[1]))
            y.append(int(row[2]))
        X = np.array(x)
        Y = np.array(y) / 20
        return X, Y

def initialize_centroids(X, Y):
    centroids_x = np.random.uniform(np.min(X), np.max(X), K_VALUE)
    centroids_y = np.random.uniform(np.min(Y), np.max(Y), K_VALUE)
    return centroids_x, centroids_y

def get_distance(point1, point2):
    a = np.array(point1)
    b = np.array(point2)
    return np.linalg.norm(a - b)

def k_means_iteration(X, Y, centroids_x, centroids_y):
    for i in range(len(X)):
        closest_k_val = 0
        closest_k_distance = sys.maxsize
        for k in range(K_VALUE):
            distance = get_distance((X[i], Y[i]), (centroids_x[k], centroids_y[k]))
            if distance < closest_k_distance:
                closest_k_distance = distance
                closest_k_val = k
        Z_VALUES[i] = closest_k_val

def update_centroids(X, Y):
    centroids_x = np.zeros(K_VALUE)
    centroids_y = np.zeros(K_VALUE)
    counts = np.zeros(K_VALUE, dtype=int)

    for i in range(len(X)):
        cluster_index = int(Z_VALUES[i])
        centroids_x[cluster_index] += X[i]
        centroids_y[cluster_index] += Y[i]
        counts[cluster_index] += 1

    for j in range(K_VALUE):
        if counts[j] > 0:
            centroids_x[j] /= counts[j]
            centroids_y[j] /= counts[j]

    return centroids_x, centroids_y

X, Y = import_data()
centroids_x, centroids_y = initialize_centroids(X, Y)
Z_VALUES = np.zeros(X.shape[0])

count = 0
while True:
    old_centroids_x, old_centroids_y = np.copy(centroids_x), np.copy(centroids_y)
    
    k_means_iteration(X, Y, centroids_x, centroids_y)
    centroids_x, centroids_y = update_centroids(X, Y)
    
    if np.all(np.abs(old_centroids_x - centroids_x) < CONVERGENCE_THRESHOLD) and \
       np.all(np.abs(old_centroids_y - centroids_y) < CONVERGENCE_THRESHOLD):
        break
    count += 1

print(f"Converged in {count} iterations")

fig, ax = plt.subplots()
scatter = ax.scatter(X, Y, c=Z_VALUES, cmap='inferno', edgecolors='black', s=50)
ax.scatter(centroids_x, centroids_y, c='red', marker='X', s=100)

plt.show()
