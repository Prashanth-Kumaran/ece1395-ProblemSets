import numpy as np
from sigmoid import sigmoid
from sklearn.preprocessing import OneHotEncoder
from sigmoidGradient import sigmoidGradient
from nnCost import nnCost
from sklearn.metrics import accuracy_score
from predict import predict

def sGD(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lamb, alpha, MaxEpochs):
    Theta1 = np.random.uniform(low=-.15, high=.15, size=(hidden_layer_size, input_layer_size + 1))
    Theta2 = np.random.uniform(low=-.15, high=.15, size=(num_labels, hidden_layer_size + 1))
    encoder = OneHotEncoder(sparse_output=False)
    y_mat = encoder.fit_transform(y_train.reshape(-1, 1))
    m = X_train.shape[0]
    costs = [0 for i in range(MaxEpochs)]
    accuracies = [0 for i in range(MaxEpochs)]
    for i in range(MaxEpochs):
        sample_permutations = np.random.permutation(m)
        for j in sample_permutations:
            # forward pass
            x = X_train[j, :].reshape(-1, 1)
            x_bias = np.insert(x, 0, 1, axis=0)
            z2 = Theta1@x_bias
            a2 = sigmoid(z2)
            a2_bias = np.insert(a2, 0, 1, axis=0)
            a3 = sigmoid(Theta2@a2_bias)

            # back propagation
            d3 = a3 - y_mat[j].reshape(-1, 1)
            d2 = (Theta2.T@d3)[1:]*sigmoidGradient(z2)

            Delta2 = d3@a2_bias.T
            Delta1 = d2@x_bias.T

            D2 = Delta2
            D2[:, 1:] = Delta2[:, 1:] + (lamb/m)*(Theta2[:, 1:])

            D1 = Delta1
            D1[:, 1:] = Delta1[:, 1:] + (lamb/m)*(Theta1[:, 1:])

            Theta2 = Theta2 - alpha*D2
            Theta1 = Theta1 - alpha*D1
        costs[i] = nnCost(Theta1, Theta2, X_train, y_train, num_labels, lamb) 
        p, h_x = predict(Theta1, Theta2, X_train)
        accuracies[i] = accuracy_score(p, y_train)
        print("Lambda = ", lamb , "Epoch ", i, "Cost: ", costs[i], "Accuracy: ", accuracy_score(p, y_train))
    p, h_x = predict(Theta1, Theta2, X_train)
    accuracy = accuracy_score(p, y_train)
    return Theta1, Theta2, costs, accuracy





def add_bias(mat):
    bias = np.ones((mat.shape[0], 1))
    return np.hstack((bias, mat))