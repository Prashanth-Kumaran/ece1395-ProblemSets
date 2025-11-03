import numpy as np
import matplotlib.pyplot as plt
from computeCost import computeCost
from gradientDescent import gradientDescent
from normalEqn import normalEqn



# //////////////// Question 1 ////////////////
print("---------------Question 1 Outputs:--------------------")
X = np.array([[1, 0, 1],
              [1, 1, 1.5],
              [1, 2, 4],
              [1, 3, 2]])

y = np.array([[1.5],
              [4],
              [8.5],
              [8.5]])

theta_1 = np.array ([[0.5],
                     [2],
                     [1]])

theta_2 = np.array ([[3],
                     [-1.5],
                     [-4]])

theta_3 = np.array ([[0.5],
                     [1],
                     [2]])

J_1 = computeCost(X, y, theta_1)
J_2 = computeCost(X, y, theta_2)
J_3 = computeCost(X, y, theta_3)




print("(i) J(θ): ", J_1[0][0], "\n(ii) J(θ): ", J_2[0][0], "\n(iii) J(θ): ", J_3[0][0])

# //////////////// Question 2 ////////////////////
print("---------------Question 2 Outputs:--------------------")
theta, cost = gradientDescent(X, y, .001, 15)

print("Gradient Descent Theta:\n", theta, "\nFinal Cost: ", cost[-1][0])

# ////////////// Question 3 ///////////////////////////
print("---------------Question 3 Outputs:--------------------")
theta = normalEqn(X, y)

print("Normal Equation Theta:\n", theta)


# ///////////// Question 4 /////////////////////////
print("---------------Question 4 Outputs:--------------------")

# 4a)
data = np.loadtxt('input/hw2_data1.csv', delimiter=',')

data_x = data[:, 0]
data_y = data[:, 1]

# 4b)
p4b1 = plt.figure()
plt.scatter(data_x, data_y, marker='x', c='red')
plt.xlabel("Horse power of a car in 100s")
plt.ylabel("Price in $1,000s")
p4b1.savefig("output/ps2-4-b.png")

# 4c)
X = np.zeros((len(data_x), 2))
X[:, 0] = 1
X[:, 1] = data_x
y = np.zeros((len(data_y), 1))
y[:, 0] = data_y

print("X size: ", X.shape, "y size: ", y.shape)


# 4d)
num_samples = len(X)


# creates a set of shuffled indices
shuffled_indices = np.random.permutation(num_samples)

# finds a split point 90% of the way through the data set, so that 90% is used for training, and 10% for testing
train_ratio = 0.9
split_point = int(num_samples * train_ratio)

# Apply shuffled indices and slice the data
X_shuffled = X[shuffled_indices]
y_shuffled = y[shuffled_indices]

X_train = X_shuffled[:split_point]
X_test = X_shuffled[split_point:]
y_train = y_shuffled[:split_point]
y_test = y_shuffled[split_point:]

# 4e)
theta, cost = gradientDescent(X_train, y_train, 0.3, 500)

iter_num = np.arange(len(cost))

p4e1 = plt.figure()
plt.plot(iter_num, cost)
plt.xlabel("Iteration #")
plt.ylabel("Cost")
p4e1.savefig("output/ps2-4-e.png")

print("Theta:\n", theta)

# 4f)
p4f1 = plt.figure()
plt.scatter(data_x, data_y, marker='x', c='red')
plt.xlabel("Horse power of a car in 100s")
plt.ylabel("Price in $1,000s")
line_x = np.linspace(min(data_x), max(data_x), 100)
line_y = theta[0] + theta[1]*line_x
plt.plot(line_x, line_y)
p4f1.savefig("output/ps2-4-f.png")


# 4g)
cost = computeCost(X_test, y_test, theta)
print("Gradient Descent Prediction Error: ", cost[0][0])

# 4h)

theta = normalEqn(X_train, y_train)
cost = computeCost(X_test, y_test, theta)
print("Normal Equation Prediction Error: ", cost[0][0])


# 4i)
p4i1 = plt.figure()
theta, cost = gradientDescent(X_train, y_train, 0.001, 300)
iter_num = np.arange(len(cost))
plt.plot(iter_num, cost, label='alpha: 0.001')
plt.xlabel("Iteration #")
plt.ylabel("Cost")
plt.legend()
p4i1.savefig("output/ps2-4-i-1.png")

p4i2 = plt.figure()
theta, cost = gradientDescent(X_train, y_train, 0.003, 300)
iter_num = np.arange(len(cost))
plt.plot(iter_num, cost, label='alpha: 0.003')
plt.xlabel("Iteration #")
plt.ylabel("Cost")
plt.legend()
p4i2.savefig("output/ps2-4-i-2.png")

p4i3 = plt.figure()
theta, cost = gradientDescent(X_train, y_train, 0.03, 300)
iter_num = np.arange(len(cost))
plt.plot(iter_num, cost, label='alpha: 0.03')
plt.xlabel("Iteration #")
plt.ylabel("Cost")
plt.legend()
p4i3.savefig("output/ps2-4-i-3.png")

p4i4 = plt.figure()
theta, cost = gradientDescent(X_train, y_train, 3, 300)
iter_num = np.arange(len(cost))
plt.plot(iter_num, cost, label='alpha: 3')
plt.xlabel("Iteration #")
plt.ylabel("Cost")
plt.legend()
p4i4.savefig("output/ps2-4-i-4.png")


# ///////////////// Question 5 //////////////////////
print("---------------Question 5 Outputs:--------------------")

# 5a)
data = np.loadtxt('input/hw2_data3.csv', delimiter=',')


data_x1 = data[:, 0]
data_x2 = data[:, 1]
data_y = data[:, 2]

x1_mean = np.mean(data_x1)
x2_mean = np.mean(data_x2)
y_mean = np.mean(data_y)

x1_stdev = np.std(data_x1)
x2_stdev = np.std(data_x2)
y_stdev = np.std(data_y)

# Standardizes the data
x1_zscore = (data_x1 - x1_mean)/x1_stdev
x2_zscore = (data_x2 - x2_mean)/x2_stdev
y_zscore = (data_y - y_mean)/y_stdev

X = np.zeros((len(x1_zscore), 3))

# Output Matrix
y = y_zscore.reshape(-1, 1)

# Constructing feature matrix
X[:, 0] = 1
X[:, 1] = x1_zscore
X[:, 2] = x2_zscore


print("x1 mean: ", x1_mean, " x2 mean: ", x2_mean, "y mean: ", y_mean)
print("x1 standard deviation: ", x1_stdev, " x2 standard deviation: ", x2_stdev, "y standard deviation: ", y_stdev)
print("X size: ", X.shape)
print("y size: ", y.shape)

# 5b)
theta, cost = gradientDescent(X, y, 0.01, 750)
p5b1 = plt.figure()
iter_num = np.arange(len(cost))
plt.plot(iter_num, cost)
plt.xlabel("Iteration #")
plt.ylabel("Cost")
p5b1.savefig("output/ps2-5-b.png")
print("Theta:\n", theta)


# 5c)

# input features to test model
x1_s = 2100
x2_s = 1200

# standardizing input features
x1_z = (x1_s - x1_mean)/x1_stdev
x2_z = (x2_s - x2_mean)/x2_stdev


y_z = theta[0] + theta[1]*x1_z + theta[2]*x2_z

# undoing standardization to get output prediction
y_pred = (y_z*y_stdev) + y_mean

print("Prediction: ", y_pred[0])