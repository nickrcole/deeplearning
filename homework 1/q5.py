import csv
import numpy as np


def get_coef(X, t):

    X_transpose = np.transpose(X)
    X_transpose_X = np.dot(X_transpose, X)
    X_transpose_X_inverse = np.linalg.inv(X_transpose_X)
    X_transpose_t = np.dot(X_transpose, t)
    w = np.dot(X_transpose_X_inverse, X_transpose_t)

    return w[0], w[1]


def main():

    x_list = []
    t_list = []

    with open('mensolympic100mdata.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            x_sub = [1, float(row[1])]
            x_list.append(x_sub)
            t_list.append(float(row[2]))

    x = np.array(x_list)
    t = np.array(t_list)

    w_0, w_1 = get_coef(x, t)
    print(f"w_0 = {w_0}\nw_1 = {w_1}")
    print("\n\nProjecting winning times up to 2008...\n\n")
    correct_times = { 1992: 9.96, 1996: 9.84,
                      2000: 9.87, 2004: 9.85, 2008: 9.69 }

    difference_sum = 0
    for year, time in correct_times.items():
        predicted_time = w_0 + w_1 * year
        difference = time - predicted_time
        print(f"Year: {year}\nPredicted time: {round(predicted_time, 4)}\nActual time: {time}\nDifference: {round(difference, 4)}\n\n")
        difference_sum += difference

    avg_difference = difference_sum / len(correct_times)
    print(f"Average error: {avg_difference}")


if __name__ == "__main__":
    main()
