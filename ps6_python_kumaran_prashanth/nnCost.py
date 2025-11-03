import numpy as np
from predict import predict
from sklearn.preprocessing import OneHotEncoder

def nnCost(Theta1, Theta2, X, y, K, lamb):
    m = len(y)
    encoder = OneHotEncoder(sparse_output=False)
    y_mat = encoder.fit_transform(y.reshape(-1, 1))
    p, h_x = predict(Theta1, Theta2, X)
    cost = (-1/m)*np.sum(y_mat * np.log(h_x) + (1-y_mat) * np.log(1-h_x))
    reg = lamb/(2*m)*(np.sum((Theta1[:, 1:])**2)+ np.sum((Theta2[:, 1:])**2))
    J = cost + reg
    return J