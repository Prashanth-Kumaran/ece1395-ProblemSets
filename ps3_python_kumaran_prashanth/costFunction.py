import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sigmoid import sigmoid

def costFunction(theta, X_train, y_train):
    m = len(y_train)
    z = X_train @ theta
    h = sigmoid(z)
    cost = -(1/m) * np.sum(y_train*np.log(h) + (1-y_train) *np.log(1-h))
    return cost

