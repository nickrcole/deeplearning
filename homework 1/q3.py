import csv
import numpy as np

def get_coef(X, t):

    XtX = np.dot(X.T, X)
    XtXi = np.linalg.inv(XtX)
    XTt = np.dot(X.T, t)
    w = np.dot(XtXi, XTt)
    return w

def main():

    x_linear = np.array([[1, -3], [1, -1], [1, 0], [1, 1], [1, 3]])
    x_quadratic = np.array([[1, -3, 9], [1, -1, 1], [1, 0, 0], [1, 1, 1], [1, 3, 9]])
    x_cubic = np.array([[1, -3, 9, -27], [1, -1, 1, -1], [1, 0, 0, 0], [1, 1, 1, 1], [1, 3, 9, 27]])
    t = np.array([3, 2, 1, -1, 4])

    w = get_coef(x_linear, t)
    print(f"Linear:\nw_0 = {w[0]}\nw_1 = {w[1]}")

    w = get_coef(x_quadratic, t)
    print(f"Quadratic:\nw_0 = {w[0]}\nw_1 = {w[1]}, w_2 = {w[2]}")

    w = get_coef(x_cubic, t)
    print(f"Cubic:\nw_0 = {w[0]}\nw_1 = {w[1]}, w_2 = {w[2]}, w_3 = {w[3]}")


if __name__ == "__main__":
    main()
