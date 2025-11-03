import scipy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random
from enum import Enum
from predict import predict
from nnCost import nnCost
from sigmoidGradient import sigmoidGradient
from sGD import sGD
class Vehicle(Enum):
    airplane = 1
    automobile = 2
    truck = 3

# 0: Data Preprossesing
data2 = scipy.io.loadmat('input/HW6_Data2_full.mat')
X = np.array(data2['X'])
y = np.array(data2['y_labels'])


sample_indeces = random.sample(range(X.shape[0]), 25);
fig, axes = plt.subplots(5, 5)
fig.tight_layout()
axes = axes.flatten()
for i, ax in enumerate(axes):
    id = sample_indeces[i]
    img = X[id, :]
    img = img.reshape(32, 32)
    label = Vehicle(y[id]).name
    ax.set_title(label)
    ax.imshow(img, cmap='gray')
plt.savefig("output/ps6-0-a-1.png")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(2/15))


# 1: Forward Propagation
weights = scipy.io.loadmat('input/HW6_weights_3_full.mat')
Theta1 = np.array(weights['Theta1'])
Theta2 = np.array(weights['Theta2'])

p, h_x = predict(Theta1, Theta2, X)

print("1b) Accuracy: ",  accuracy_score(p, y))


# 2: Cost Function
print("2b) Cost:")
lamb = [0.1, 1, 2]
J = [0 for i in range(len(lamb))]
for i in range(len(lamb)):
    J[i] = nnCost(Theta1, Theta2, X, y, 3, lamb[i])
    print('lambda = ' , lamb[i], ':' , 'J = ', J[i])


# 3: Sigmoid Gradient
z = np.array([[-10],[0],[10]])
g_prime = sigmoidGradient(z)
print("3) Gradient = ", g_prime)


# 4: Back Propagation
lambdas = [0.01, 0.1, 0.2, 1]
for lamb in lambdas:
    Theta1, Theta2, cost, accuracy = sGD(1024, 40, 3, X_train, y_train, lamb, 0.001, 50)
    p, h_x = predict(Theta1, Theta2, X_test)
    testCost = nnCost(Theta1, Theta2, X_test, y_test, 3, lamb)
    print("Lambda = ", lamb, "Testing Accuracy = ",  accuracy_score(p, y_test), "Testing Cost = ", testCost)