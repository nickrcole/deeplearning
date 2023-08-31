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

    x = np.array([])
    t = np.array([])

    with open('mensolympic100mdata.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            x = np.append(x, float(row[1]))
            t = np.append(t, float(row[2]))

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
