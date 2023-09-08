import csv
import numpy as np

def get_coef(x, t):
    n = np.size(x)

    x_avg = np.mean(x)
    t_avg = np.mean(t)

    numerator = ( np.sum(x*t) / n ) - x_avg * t_avg
    denominator = ( np.sum(x*x) / n ) - x_avg * x_avg

    w_1 = numerator / denominator
    w_0 = t_avg - w_1*x_avg

    return w_0, w_1


def main():

    X_list = []
    y_list = []

    with open('heightandweight.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            X_list.append([float(row[0])])
            y_list.append([float(row[1])])

    x = np.array(X_list)
    t = np.array(y_list)

    w_0, w_1 = get_coef(x, t)
    print(f"w_0 = {w_0}\nw_1 = {w_1}")

    sum = 0
    for i in range(0, len(x)):
        sum += (t[i] - (x[i] * w_1 + w_0))**2
    r2 = sum / len(t)
    print(r2)


if __name__ == "__main__":
    main()
