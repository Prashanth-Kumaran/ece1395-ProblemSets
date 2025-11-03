import numpy as np
import matplotlib.pyplot as plt
import time

# ////////////////// Question 3 /////////////////
print("---------------Question 3 Text Outputs-----------------")
# 3a)
x_mean = 2.4
x_stdev = 0.75
vec_size = 1000000

x_vector = np.random.normal(loc=x_mean, scale=x_stdev, size=vec_size)

# 3b)
z_lower = -2
z_upper = 1

z_vector = np.random.uniform(low=z_lower, high=z_upper, size=vec_size)

# 3c)
figx = plt.figure()
plt.hist(x_vector, bins=30, density=True)
plt.xlabel("X Values")
plt.ylabel("Frequency")
plt.title("Vector X Histogram")
figx.savefig('output/ps1-3-c-1.png')

figz = plt.figure()
plt.hist(z_vector, bins=30, density=True)
plt.xlabel("Z Values")
plt.ylabel("Frequency")
plt.title("Vector Z Histogram")
figz.savefig('output/ps1-3-c-2.png')

# 3d)
start_time = time.time()
for i in range(x_vector.shape[0]):
    x_vector[i] = x_vector[i] + 2
end_time = time.time()
exec_time = end_time - start_time
print(f"Loop execution time: {exec_time:.4f} seconds")

# 3e)
start_time = time.time()
np.add(x_vector, 2)
end_time = time.time()
exec_time = end_time - start_time
print(f"Optimized execution time: {exec_time:.4f} seconds")

# 3f)
y_indices = np.where((z_vector > 0) & (z_vector < 0.8))
y_vector = z_vector[y_indices]
print(f"Vector y size : {y_vector.shape[0]}")

# /////////////////// Question 4 //////////////////
print("---------------Question 4 Text Outputs-----------------")
# 4a)
A = np.array([[2, 10, 8],
              [3, 5, 2],
              [6, 4, 4]])

# minimum in each column
min_cols = np.min(A, axis=0)
print("Min in each column: ", min_cols)
# max in each row
max_rows = np.max(A, axis=1)
print("Max in each row: ", max_rows)
# sum of each row
sum_rows = np.sum(A, axis=1)
print("Sum of each row: ", sum_rows)
# sum of all elements
total_sum = np.sum(A)
print("Sum of all elements: ", total_sum)

# square every element
B = np.square(A)
print("Matrix B: ", B)


# 4b)
# coefficient matrix A
A = np.array([[2, 5, -2],
              [2, 6, 4],
              [6, 8, 18]])

# constant vector b
b = np.array([12, 6, 15])

A_inv = np.linalg.inv(A)

x = np.matmul(A_inv, b)

print("Resultant: ", x)



# 4c)
x1 = np.array([-4, 0, 1])
x2 = np.array([-2, -2, 0])

x1_normL1 = np.linalg.norm(x1, ord=1)
x1_normL2 = np.linalg.norm(x1, ord=2)

x2_normL1 = np.linalg.norm(x2, ord=1)
x2_normL2 = np.linalg.norm(x2, ord=2)


print(f"x1 L1 norm: {x1_normL1}")
print(f"x1 L2 norm: {x1_normL2}")
print(f"x2 L1 norm: {x2_normL1}")
print(f"x2 L2 norm: {x2_normL2}")




# ///////////////////////// Question 5 /////////////////////////
print("---------------Question 5 Text Outputs-----------------")

# 5a)
X = np.zeros((10, 3), dtype=int)

for i in range(10):
    X[i][:] = i

y = np.arange(1, 11)

print("Matrix X:\n", X)

# 5b)

np.random.shuffle(X)

X_train = X[:8]
X_test = X[8:]
print("X_train:\n", X_train)
print("X_test:\n", X_test)

# 5c)
y_train = np.zeros(8, dtype=int)
y_test = np.zeros(2, dtype=int)
for i in range(8):
    y_train[i] = y[X_train[i][0]]-1

y_test[0] = y[X_test[0][0]]-1
y_test[1] = y[X_test[1][0]]-1

print("y_train:\n", y_train)
print("y_test:\n", y_test)