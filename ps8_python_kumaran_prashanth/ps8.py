import scipy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import random

# 1. Bagging and Digit Recognition
data1 = scipy.io.loadmat('input/HW8_data1.mat')
X = np.array(data1['X'])
y = np.array(data1['y'])

# a)
indeces = random.sample(range(len(y)), 25)
fig, axs = plt.subplots(nrows = 4, ncols=5)
for idx, ax in enumerate(axs.flatten()):
    img = X[indeces[idx]].reshape(20, 20)
    ax.imshow(img, cmap='gray')
plt.savefig('output/ps8-1-a-1.png')

# b)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.14)

y = y.ravel()
y_train = y_train.ravel()
y_test = y_test.ravel()

# c) 
X_idx = [random.sample(range(len(y_train)), 1250) for i in range(5)]
Xi = [X_train[X_idx[i], :] for i in range(5)]
yi = [y_train[X_idx[i]] for i in range(5)]

# d)
svm = SVC(kernel='rbf', decision_function_shape='ovr')
svm.fit(Xi[0], yi[0])

# i. training error on X1
pred_train = svm.predict(Xi[0])
err_train = 1 - accuracy_score(yi[0], pred_train)

# ii. errors on X2 to X5
errors_other = []
for i in range(1, 5):
    pred = svm.predict(Xi[i])
    err = 1 - accuracy_score(yi[i], pred)
    errors_other.append(err)

# iii. error on testing set
pred_test = svm.predict(X_test)
err_test = 1 - accuracy_score(y_test, pred_test)

print("Training error (X1):", err_train)
print("Other subset errors (X2–X5):", errors_other)
print("Testing error:", err_test)

