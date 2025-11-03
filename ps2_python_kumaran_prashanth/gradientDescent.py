import numpy as np
import matplotlib.pyplot as plt
from computeCost import computeCost

def gradientDescent(X_train, y_train, alpha, iters):
    n = X_train.shape[1] - 1
    m = y_train.shape[0]

    # starting with random theta
    theta = np.random.rand(n+1, 1)
    cost = np.zeros((iters, 1))


    for i in range(iters):
        h = (X_train@theta)
        error = h - y_train
        gradient = (1/m) * X_train.T@error
        theta = theta - alpha * gradient
        cost[i] = computeCost(X_train, y_train, theta)

    return theta, cost
    
    