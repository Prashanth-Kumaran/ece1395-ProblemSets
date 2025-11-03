import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sigmoid import sigmoid
from costFunction import costFunction
from gradFunction import gradFunction
from scipy.optimize import fmin_bfgs
from normalEqn import normalEqn
# ////////////// Question 1 ////////////////////
print("---------------Question 1 Outputs:--------------------")
# 1a)
# Loading Data
data = np.loadtxt('input/hw3_data1.txt', delimiter=',')

data_x = data[:, 0:2]
data_y = data[:, 2]

# Forming Feature and Output Matrices
X = np.zeros((len(data_x), 3))
X[:, 0] = 1
X[:, 1] = data_x[:, 0]
X[:, 2] = data_x[:, 1]

y = np.zeros((len(data_y), 1))
y = data_y

print("X Shape:", X.shape)
print("y shape:", y.shape)

# 1b)
p1b1 = plt.figure()
plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y)
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.title('ps3-1-b.png')
p1b1.savefig('output/ps3-1-b.png')

# 1c)
# Splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# 1d)
# Testing Sigmoid Function
z = np.arange(-15, 15, 0.01)
gz = sigmoid(z)

p1c1 = plt.figure()
plt.plot(z, gz)
p1c1.savefig('output/ps3-1-c.png')

# 1e)
# Testing Logistic Regression on Toy Data Set
toy_X = np.array([[1, 1, 0], [1, 1, 3], [1, 3, 1], [1, 3, 4]])
toy_y = np.array([[0], [1], [0], [1]])
toy_theta = np.array([[1], [0.9], [1.2]])


cost = costFunction(toy_theta, toy_X, toy_y)
gradient = gradFunction(toy_theta, toy_X, toy_y)
print("Cost:", cost)
# print("Gradient: ", gradient)

# 1f)
# Optimizing Logistic Regression functions to find optimal theta
initial_theta = np.zeros(X_train.shape[1])
optimal_theta = fmin_bfgs(costFunction, initial_theta, fprime=gradFunction, args=(X_train, y_train), maxiter=400)

print("Optimized Theta: ", optimal_theta)
cost = costFunction(optimal_theta, X_train, y_train)
print("Optimized Cost: ", cost)


# 1g) 

p1g1 = plt.figure()
plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y)
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
x1_values = np.linspace(20, 100, 1000)
x2_values = -(optimal_theta[0] + optimal_theta[1]*x1_values) / optimal_theta[2]
plt.plot(x1_values, x2_values, label='Decision Boundary', color='red')
plt.legend()
p1g1.savefig("output/ps3-1-g.png")

# 1h)

# Finding predicted values based on model, then computing accuracy
y_pred = np.round(sigmoid(X_test @ optimal_theta))
accuracy = np.mean(y_test == y_pred)

print('Accuracy: ', accuracy*100, '%')


# 1i)
# Finding the probability and classification of a data point based on the model
prob = sigmoid(optimal_theta[0] + optimal_theta[1] * 55 + optimal_theta[2] * 70)
print('Probability of Student Acceptance: ', prob)

if prob> 0.5:
    print('Student Admitted')
else:
    print('Student Rejected')


# ////////////// Question 2 ////////////////////
print("---------------Question 2 Outputs:--------------------")
data = np.loadtxt('input/hw3_data2.csv', delimiter=',')

data_x = data[:, 0]
data_y = data[:, 1]

X = np.zeros((len(data_x), 3))
X[:, 0] = 1
X[:, 1] = data_x
X[:, 2] = (data_x)**2

y = data_y

theta = normalEqn(X, y)

print(theta)

p2b1 = plt.figure()
plt.scatter(data_x, data_y, marker='x', c='red')
plt.xlabel("population in 1000s")
plt.ylabel("profit")
line_x = np.linspace(min(data_x), max(data_x), 1000)
line_y = theta[0] + theta[1]*line_x + theta[2]*(line_x**2)
plt.plot(line_x, line_y)
p2b1.savefig("output/ps3-2-b.png")