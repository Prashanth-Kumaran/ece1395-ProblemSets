import numpy as np
from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
    X_bias = add_bias(X)
    a1 = sigmoid(X_bias@Theta1.T)
    a1_bias = add_bias(a1)
    h_x = sigmoid(a1_bias@Theta2.T)
    p = np.argmax(h_x, axis=1) + 1

    return p, h_x

def add_bias(mat):
    bias = np.ones((mat.shape[0], 1))
    return np.hstack((bias, mat))