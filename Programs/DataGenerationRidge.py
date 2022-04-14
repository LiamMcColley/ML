# Machine Learning HW2 Data Generation

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
        print("Usage \"python3 DataGenerationRidge.py 200\"")
        sys.exit()
    x = np.random.uniform(-10, 10, num_to_generate) # generate x values
    garbage_xs = np.random.uniform(0.0, 0.3, (num_to_generate, 100)) # generate meaningless features
    y = 5*x + 3 + np.random.normal(0.0, 2, num_to_generate) # using an underlying "true" data distribution and some guassian noise
    file = open("dataRidge.txt","w") # write x and y tab seperated to a file
    for i in range(num_to_generate):
        file.write("1" + "\t" + str(x[i]) + "\t")
        for j in range(100):
            file.write(str(garbage_xs[i,j]) + "\t")
        file.write(str(str(y[i])) + "\n")
