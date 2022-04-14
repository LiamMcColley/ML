
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv


# Parse the file and return 2 numpy arrays


def load_data_set(filename):
    data = np.loadtxt(filename)
    x = data[:, (0, 1)]
    y = data[:, 2]
    return x, y

# Find theta using the normal equation


def normal_equation(x, y):
    xt = np.transpose(x)
    xmul = np.matmul(xt, x)
    xinv = inv(xmul)
    xmul2 = np.matmul(xinv, xt)
    theta = np.matmul(xmul2, y)
    return theta

# Find thetas using stochastic gradient descent
# Don't forget to shuffle


def stochastic_gradient_descent(x, y, learning_rate, num_epoch):

    thetas = [np.array([np.random.normal(), np.random.normal()])]
    for i in range(num_epoch-1):
        shuffler = np.random.permutation(len(x))
        x = x[shuffler]
        y = y[shuffler]
        step0 = np.matmul(np.transpose(x[i]), thetas[i])
        step = (y[i] - step0) * x[i]
        update = thetas[i] + (learning_rate * step)
        thetas.append(update)
    return thetas

# Find thetas using gradient descent


def gradient_descent(x, y, learning_rate, num_epoch):

    n = y.size
    thetas = [np.array([np.random.normal(), np.random.normal()])]
    for i in range(num_epoch-1):
        step0 = np.matmul(x, thetas[i])
        step = np.matmul(np.transpose(x), (y - step0))
        update = thetas[i] + (learning_rate * step)/n
        thetas.append(update)
    return thetas

# Find thetas using minibatch gradient descent after shuffling


def minibatch_gradient_descent(x, y, learning_rate, num_epoch, batch_size):
    thetas = [np.array([np.random.normal(), np.random.normal()])]
    shuffler = np.random.permutation(len(x))
    x = x[shuffler]
    y = y[shuffler]
    for i in range(num_epoch-1):
        step1 = [0, 0]
        for j in range(batch_size):
            step0 = np.matmul(np.transpose(x[i+j]), thetas[i])
            step = (y[i+j] - step0) * x[i+j]
            step1 += step
        update = thetas[i] + (learning_rate * step1)/batch_size
        thetas.append(update)
    return thetas

# Given an array of x and theta predict y


def predict(x, theta):
    y_predict = np.matmul(x, theta)
    return y_predict

# Given an array of y and y_predict return MSE loss


def get_mseloss(y, y_predict):
    n = y.size
    x = 0
    for i in range(n):
        x += (y[i]-y_predict[i])**2
    return x/n

# Given a list of thetas one per epoch
# this creates a plot of epoch vs training error


def plot_training_errors(x, y, thetas, title):
    losses = []
    epochs = []
    losses = []
    epoch_num = 1
    for theta in thetas:
        losses.append(get_mseloss(y, predict(x, theta)))
        epochs.append(epoch_num)
        epoch_num += 1
    plt.plot(epochs, losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.show()

# Given x, y, y_predict and title,
# this creates a plot


def plot(x, y, theta, title):
    y_predict = predict(x, theta)
    plt.scatter(x[:, 1], y)
    plt.plot(x[:, 1], y_predict)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    # first column in data represents the intercept term, second is the x value, third column is y value
    x, y = load_data_set('regression-data.txt')
    plt.scatter(x[:, 1], y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Scatter Plot of Data")
    plt.show()
    theta = normal_equation(x, y)
    plot(x, y, theta, "Normal Equation Best Fit")

    # thetas records the history of (s)GD optimization e.g. thetas[epoch] with epoch=0,1,,......T
    thetas = gradient_descent(x, y, 0.3, 100)
    print(thetas[-1])
    plot(x, y, thetas[-1], "Gradient Descent Best Fit")
    plot_training_errors(
        x, y, thetas, "Gradient Descent Epoch vs Mean Training Loss")

    thetas = stochastic_gradient_descent(x, y, 0.4, 100)
    print(thetas[-1])
    plot(x, y, thetas[-1], "stochastic Gradient Descent Best Fit")
    plot_training_errors(
        x, y, thetas, "Stochastic Gradient Descent Epoch vs Mean Training Loss")

    thetas = minibatch_gradient_descent(x, y, 1, 100, 5)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit")
    plot_training_errors(
        x, y, thetas, "Minibatch Gradient Descent Epoch vs Mean Training Loss")

    thetas = minibatch_gradient_descent(x, y, 1, 100, 40)
    print(thetas[-1])
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit")
    plot_training_errors(
        x, y, thetas, "Minibatch Gradient Descent Epoch vs Mean Training Loss")

    thetas = minibatch_gradient_descent(x, y, 1, 100, 100)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit")
    plot_training_errors(
        x, y, thetas, "Minibatch Gradient Descent Epoch vs Mean Training Loss")
