# Machine Learning HW2 Data Generation
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import sys
np.random.seed(37)
# Given x, y this creates a plot


def plot(x, y):
    # plot
    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        num_to_generate = int(sys.argv[1])
    else:
        print("Usage \"python3 DataGenerationPoly.py 300\"")
        sys.exit()
    x = np.random.uniform(-10, 10, num_to_generate)  # generate x value
    # using an underlying "true" data distribution and some gaussian noise this generates ys given xs
    y = 0.05*np.power(x, 5) - 5*np.power(x, 3) + 10*np.power(x, 2) - \
        5*x - 39 + np.random.normal(0.0, 100, num_to_generate)

    # scale x features
    scx = preprocessing.StandardScaler()
    x = x.reshape((x.shape[0], 1))
    x = scx.fit_transform(x)
    x = x.reshape((x.shape[0]))

    # scale y features also
    y = y.reshape((y.shape[0], 1))
    y = scx.fit_transform(y)
    y = y.reshape((y.shape[0]))

    plot(x, y)

    file = open("dataPoly.txt", "w")  # write x and y tab seperated to a file
    for elem in list(zip(x, y)):
        file.write(str(elem[0]) + "\t" + str(elem[1]) + "\n")
