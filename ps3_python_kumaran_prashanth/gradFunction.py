import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sigmoid import sigmoid

def gradFunction(theta, X_train, y_train):
    m = len(y_train)
    h = sigmoid(X_train @ theta)
    gradient = (1/m) * (X_train.T @ (h - y_train))
    return gradient