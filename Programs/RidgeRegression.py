#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 17:50:41 2020

@author: zhe
"""

# Machine Learning HW2 Ridge Regression

from cmath import sqrt
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm

# Parse the file and return 2 numpy arrays


def load_data_set(filename):
    x = np.loadtxt(filename, usecols=(range(102)))
    y = np.loadtxt(filename, usecols=102)
    return x, y


# Split the data into train and test examples by the train_proportion
# i.e. if train_proportion = 0.8 then 80% of the examples are training and 20%
# are testing
def train_test_split(x, y, train_proportion):
    # your code
    t = np.size(y)*train_proportion
    t = int(t)
    x_train = x[:t]
    x_test = x[t:]
    y_train = y[:t]
    y_test = y[t:]
    return x_train, x_test, y_train, y_test

# Find theta using the modified normal equation, check our lecture slides
# Note: lambdaV is used instead of lambda because lambda is a reserved word in python


def normal_equation(x, y, lambdaV):
    # your code
    i = np.identity(x.shape[1])
    xt = np.transpose(x)
    xtx = np.matmul(xt, x)
    lambdai = lambdaV*i
    xty = np.matmul(xt, y)
    beta = np.matmul(inv(xtx + lambdai), xty)
    return beta


# Given an array of y and y_predict return loss
def get_loss(y, y_predict):
    # your code
    n = y.size
    x = 0
    for i in range(n):
        x += (y[i]-y_predict[i])**2
    loss = x/n
    return loss

# Given an array of x and theta predict y


def predict(x, theta):
    # your code
    y_predict = np.matmul(x, theta)
    return y_predict

# Find the best lambda given x_train and y_train using 4 fold cv


def cross_validation(x_train, y_train, lambdas):
    valid_losses = []
    training_losses = []
    # your code
    split = len(x_train)/4
    split = int(split)
    for l in lambdas:
        tempvalidloss = []
        temptrainloss = []
        for i in range(4):

            xvalid = x_train[i*split:(i+1)*split, :]
            xtrain = np.concatenate(
                (x_train[:i*split, :], x_train[(i+1)*split:, :]), axis=0)
            yvalid = y_train[i*split:(i+1)*split]
            ytrain = np.concatenate(
                (y_train[:i*split], y_train[(i+1)*split:]), axis=0)

            beta = normal_equation(xtrain, ytrain, l)
            ytrainpred = predict(xtrain, beta)
            yvalpred = predict(xvalid, beta)

            trainloss = get_loss(ytrain, ytrainpred)
            validloss = get_loss(yvalid, yvalpred)
            tempvalidloss.append(validloss)
            temptrainloss.append(trainloss)
        avgtrainloss = np.mean(temptrainloss)
        training_losses.append(avgtrainloss)
        avgvalidloss = np.mean(tempvalidloss)
        valid_losses.append(avgvalidloss)
    return np.array(valid_losses), np.array(training_losses)


# Calcuate the l2 norm of a vector
def l2norm(vec):
    # your code

    #norm = np.linalg.norm(vec)
    sum = 0
    for x in vec:
        sum += (x**2)
    norm = sqrt(sum)
    return norm

#  show the learnt values of Î² vector from the best Î»


def bar_plot(beta):
    x = np.arange(len(beta))
    y = beta.reshape(len(beta),)
    plot = plt.figure()
    axis = plot.add_axes([0, 0, 1, 1])
    axis.bar(x, y)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.show()


if __name__ == "__main__":

    # step 1
    # If we don't have enough data we will use cross validation to tune hyperparameter
    # instead of a training set and a validation set.
    x, y = load_data_set("dataRidge.txt")  # load data
    x_train, x_test, y_train, y_test = train_test_split(x, y, 0.8)
    # Create a list of lambdas to try when hyperparameter tuning
    lambdas = [2**i for i in range(-3, 9)]
    lambdas.insert(0, 0)
    # Cross validate
    valid_losses, training_losses = cross_validation(x_train, y_train, lambdas)
    # Plot training vs validation loss
    plt.plot(lambdas[1:], training_losses[1:], label="training_loss")
    # exclude the first point because it messes with the x scale
    plt.plot(lambdas[1:], valid_losses[1:], label="validation_loss")
    plt.legend(loc='best')
    plt.xscale("log")
    plt.yscale("log")
    plt.title("lambda vs training and validation loss")
    plt.xlabel("Lambda")
    plt.ylabel("Loss")
    plt.show()

    best_lambda = lambdas[np.argmin(valid_losses)]

    # step 2: analysis
    normal_beta = normal_equation(x_train, y_train, 0)
    best_beta = normal_equation(x_train, y_train, best_lambda)
    large_lambda_beta = normal_equation(x_train, y_train, 512)
    # your code get l2 norm of normal_beta
    normal_beta_norm = l2norm(normal_beta)
    best_beta_norm = l2norm(best_beta)  # your code get l2 norm of best_beta
    # your code get l2 norm of large_lambda_beta
    large_lambda_norm = l2norm(large_lambda_beta)
    print(best_lambda)
    print("L2 norm of normal beta:  " + str(normal_beta_norm))
    print("L2 norm of best beta:  " + str(best_beta_norm))
    print("L2 norm of large lambda beta:  " + str(large_lambda_norm))
    print("Average testing loss for normal beta:  " +
          str(get_loss(y_test, predict(x_test, normal_beta))))
    print("Average testing loss for best beta:  " +
          str(get_loss(y_test, predict(x_test, best_beta))))
    print("Average testing loss for large lambda beta:  " +
          str(get_loss(y_test, predict(x_test, large_lambda_beta))))

    # step 3: visualization
    bar_plot(best_beta)
