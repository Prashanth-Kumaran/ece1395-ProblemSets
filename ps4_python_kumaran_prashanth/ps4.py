import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from Reg_normalEqn import Reg_normalEqn
from computeCost import computeCost
from logReg_multi import logReg_multi
from sklearn.neighbors import KNeighborsClassifier
# ////////////////// Question 1 /////////////////////

# 1b)
data1 = scipy.io.loadmat('input/hw4_data1.mat')
X = np.array(data1["X_data"])
y = np.array(data1["y"])
bias_column = np.ones((X.shape[0], 1))
X = np.hstack((bias_column, X))
print("1b) Feature Matrix Size: ", X.shape)

# 1c)
lamb_array = [0, 0.001, 0.003, 0.005, 0.007, 0.009, 0.012, 0.017]
train_error_mat = np.zeros((20, 8))
test_error_mat = np.zeros((20, 8))

for i in range(20):
    # i) Splitting Data into Test and Train
    num_samples = len(X)
    shuffled_indices = np.random.permutation(num_samples)
    train_ratio = 0.85
    split_point = int(num_samples * train_ratio)
    X_shuffled = X[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    X_train = X_shuffled[:split_point]
    X_test = X_shuffled[split_point:]
    y_train = y_shuffled[:split_point]
    y_test = y_shuffled[split_point:]

    # ii) Creating models for each lambda
    for j in range(8):
        theta = Reg_normalEqn(X_train, y_train, lamb_array[j])

    # iii) Computing Error for each theta
        train_error_mat[i][j] = computeCost(X_train, y_train, theta)
        test_error_mat[i][j] = computeCost(X_test, y_test, theta)

avg_train_err = np.mean(train_error_mat, axis=0)
avg_test_err = np.mean(test_error_mat, axis=0)


p1c1 = plt.figure()
plt.xlabel('Lambda')
plt.ylabel('Average Error')
plt.plot(lamb_array, avg_test_err, color='b', marker='o', label='test')
plt.plot(lamb_array, avg_train_err, color='r', marker='x', label='train')
plt.legend()
p1c1.savefig("output/ps4-1-a.png")

# /////////////// Question 2 ///////////////////

data2 = scipy.io.loadmat('input/hw4_data2.mat')
# print(data2.keys())
X1 = np.array(data2['X1'])
X2 = np.array(data2['X2'])
X3 = np.array(data2['X3'])
X4 = np.array(data2['X4'])
X5 = np.array(data2['X5'])
y1 = np.array(data2['y1'])
y2 = np.array(data2['y2'])
y3 = np.array(data2['y3'])
y4 = np.array(data2['y4'])
y5 = np.array(data2['y5'])


X_train1 = np.vstack((X1, X2, X3, X4))
y_train1 = np.vstack((y1, y2, y3, y4))

X_train2 = np.vstack((X1, X2, X3, X5))
y_train2 = np.vstack((y1, y2, y3, y5))

X_train3 = np.vstack((X1, X2, X4, X5))
y_train3 = np.vstack((y1, y2, y4, y5))

X_train4 = np.vstack((X1, X3, X4, X5))
y_train4 = np.vstack((y1, y3, y4, y5))

X_train5 = np.vstack((X2, X3, X4, X5))
y_train5 = np.vstack((y2, y3, y4, y5))

K_val = list(range(1, 16, 2))
acc_val = []
for K in K_val:
    fold1 = KNeighborsClassifier(n_neighbors=K)
    fold2 = KNeighborsClassifier(n_neighbors=K)
    fold3 = KNeighborsClassifier(n_neighbors=K)
    fold4 = KNeighborsClassifier(n_neighbors=K)
    fold5 = KNeighborsClassifier(n_neighbors=K)

    fold1.fit(X_train1, y_train1.ravel())
    fold2.fit(X_train2, y_train2.ravel())
    fold3.fit(X_train3, y_train3.ravel())
    fold4.fit(X_train4, y_train4.ravel())
    fold5.fit(X_train5, y_train5.ravel())

    acc1 = fold1.score(X5, y5)
    acc2 = fold2.score(X4, y4)
    acc3 = fold3.score(X3, y3)
    acc4 = fold4.score(X2, y2)
    acc5 = fold5.score(X1, y1)

    avg_acc = (acc1 + acc2 + acc3 + acc4 + acc5) / 5
    acc_val.append(avg_acc)
    
p2a1 = plt.figure()
plt.xlabel('K')
plt.ylabel('Average Accuracy')
plt.plot(K_val, acc_val, color='b', marker='o')

p2a1.savefig("output/ps4-2-a.png")

# ///////////////// Question 3 /////////////////////

data3 = scipy.io.loadmat('input/hw4_data3.mat')

X_test = np.array(data3['X_test'])
X_train = np.array(data3['X_train'])
y_train = np.array(data3['y_train'])
y_test = np.array(data3['y_test'])

# Compute predictions
y_train_pred = logReg_multi(X_train, y_train, X_train)
y_test_pred  = logReg_multi(X_train, y_train, X_test)

# Flatten to 1D for comparison
y_train_true = y_train.ravel()
y_test_true  = y_test.ravel()
y_train_pred = y_train_pred.ravel()
y_test_pred  = y_test_pred.ravel()

# Training and testing accuracy
train_acc = np.mean(y_train_pred == y_train_true)
test_acc  = np.mean(y_test_pred == y_test_true)

print(f"3b) Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy:  {test_acc:.4f}")