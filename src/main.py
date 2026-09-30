import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.loss import mae_


def zscore(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x' as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
    This function shouldn't raise any Exception.
    """

    m = x.shape[0]
    mean = 0
    for i in range(m):
        mean += x[i]

    mean = mean / m if m > 0 else 0

    std = 0
    for i in range(m):
        std += (x[i] - mean) ** 2
    std /= m if m > 0 else 0
    std = np.sqrt(std)

    x_normalized = (x - mean) / np.where(std == 0, 1, std)

    return x_normalized


def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
    The three arrays must have the compatible dimensions.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
        The gradient as a numpy.array, a vector of dimensions n * 1,
        containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible dimensions.
        None if x, y or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    m = x.shape[0]
    X = np.hstack((np.ones((m, 1)), x))

    errors = X @ theta - y
    grad = X.T @ errors / m

    return grad


def predict_(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of dimensions m * n.
        theta: has to be an numpy.array, a vector of dimensions (n + 1) * 1.
    Return:
        y_hat as a numpy.array, a vector of dimensions m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
        None if x or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    m = x.shape[0]
    X = np.hstack([np.ones((m, 1)), x])
    y_hat = X @ theta

    return y_hat


class MyLinearRegression:
    """
    Description:
    Linear Regression class
    """

    def __init__(self, thetas=[0, 0], alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.asarray(thetas, dtype=float).reshape(-1, 1)
        self.costs = []

    def fit_(self, x, y):
        self.costs.clear()

        new_theta = self.thetas.copy()
        for _ in range(self.max_iter):
            y_hat = predict_(x, new_theta)
            current_loss = self.loss_(y, y_hat)
            self.costs.append(current_loss)

            grad = gradient(x, y, new_theta)
            new_theta = new_theta - self.alpha * grad

        self.thetas = new_theta
        return self

    def plot_learning_curve(self):
        plt.figure()
        plt.xlabel("iterations")
        plt.ylabel("J(w, b)")

        plt.plot(self.costs, label="learning curve")
        plt.legend()
        plt.show()

    def predict_(self, x):
        return predict_(x, self.thetas)

    def loss_(self, y, y_hat):
        return mae_(y, y_hat)
