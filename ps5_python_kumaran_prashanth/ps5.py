import numpy as np
import matplotlib.pyplot as plt
import sklearn
import scipy
from weightedKNN import weightedKNN
import os
import shutil
import random
from pathlib import Path
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
import time

# ////////// Question 1 ///////////////
data3 = scipy.io.loadmat('input/hw4_data3.mat')

X_test = np.array(data3['X_test'])
X_train = np.array(data3['X_train'])
y_train = np.array(data3['y_train'])
y_test = np.array(data3['y_test'])

sigmas = [0.01, 0.07, 0.15, 1.5, 2, 4]
accuracy = []
for s in sigmas:
    y_prediction = weightedKNN(X_train, y_train, X_test, s)
    accuracy.append(float(np.mean(y_test.flatten() == y_prediction)))
    

print("sigmas: ", sigmas)
print("accuracy: ", accuracy)

# ///////////// Question 2 ///////////////

# 2.0 Data Preprocessing:

# Define paths
input_base = Path("input/all")
train_dir = Path("input/train")
test_dir = Path("input/test")

# Create output directories, removing old contents if they exist
if train_dir.exists():
    shutil.rmtree(train_dir)
if test_dir.exists():
    shutil.rmtree(test_dir)

train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

# Process each folder s1 to s40
for i in range(1, 41):
    folder_name = f"s{i}"
    folder_path = input_base / folder_name
    
    if not folder_path.exists():
        print(f"Warning: {folder_path} does not exist, skipping...")
        continue
    
    # Get all .pgm files in the folder
    images = [f for f in os.listdir(folder_path) if f.endswith('.pgm')]
    
    if len(images) != 10:
        print(f"Warning: {folder_name} has {len(images)} images instead of 10")
    
    # Randomly select 8 for training and 2 for testing
    random.shuffle(images)
    train_images = images[:8]
    test_images = images[8:]
    
    # Copy training images
    for img in train_images:
        src = folder_path / img
        # Extract original index from filename (e.g., "1.pgm" -> "1")
        orig_idx = img.replace('.pgm', '')
        dst = train_dir / f"{folder_name}_{orig_idx}.pgm"
        shutil.copy2(src, dst)
    
    # Copy testing images
    for img in test_images:
        src = folder_path / img
        # Extract original index from filename (e.g., "1.pgm" -> "1")
        orig_idx = img.replace('.pgm', '')
        dst = test_dir / f"{folder_name}_{orig_idx}.pgm"
        shutil.copy2(src, dst)

train_images = [f for f in os.listdir(train_dir) if f.endswith('.pgm')]
random_sample = random.sample(range(320), 3)

train_random_sample = [train_images[i] for i in random_sample]


fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, img_name in zip(axes, train_random_sample):
    img_path = train_dir / img_name
    img = plt.imread(img_path)
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(img_name)

plt.savefig("output/ps5-2-0.png")

# 2.1 PCA Analysis:

# 2.1 a)
T = np.zeros((10304, 320))
for i, img_name in enumerate(train_images):
    img_path = os.path.join(train_dir, img_name)
    img_array = np.array(Image.open(img_path))
    T[:, i] = img_array.ravel()

T_image = Image.fromarray(T)

T_normalized = ((T - T.min()) / (T.max() - T.min()) * 255).astype(np.uint8)

# Save as PNG
Image.fromarray(T_normalized).save("output/ps5-1-a.png")

# 2.1 b)
mean_vector = np.mean(T, axis=1)
resized_mean_vector = mean_vector.reshape(112, 92)
rmv_normalized = ((resized_mean_vector - resized_mean_vector.min()) / (resized_mean_vector.max() - resized_mean_vector.min()) * 255).astype(np.uint8)
Image.fromarray(rmv_normalized).save("output/ps5-2-1-b.png")

# 2.1 c)
m = mean_vector.reshape(-1, 1)
A = T - m
C = A@A.T

C_normalized = ((C - C.min()) / (C.max() - C.min()) * 255).astype(np.uint8)
Image.fromarray(C_normalized).save("output/ps5-2-1-c.png")


eigenvalues, eigenvectors = np.linalg.eig(A.T@A)
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

k = np.arange(len(eigenvalues))
v_k = [np.sum(eigenvalues[0:k])/np.sum(eigenvalues) for k in range(len(eigenvalues))]

p521d = plt.figure()
plt.xlabel("k")
plt.ylabel("V(k)")
plt.plot(k, v_k)
p521d.savefig("output/ps5-2-1-d.png")

v_k = np.array(v_k)
K = np.argmax(v_k > 0.95)


w, U = scipy.sparse.linalg.eigsh(C, k=K, which='LM')
print("U Dimensions: ", U.shape)

fig, axes = plt.subplots(3, 3, figsize=(9, 9))

for i, ax in enumerate(axes.flat):
    eigenface = U[:, i].real.reshape(112, 92)
    normalized_eigenface = ((eigenface - eigenface.min()) / (eigenface.max() - eigenface.min()) * 255).astype(np.uint8)
    ax.imshow(normalized_eigenface, cmap='gray')
    ax.axis('off')
    ax.set_title(f"Eigenface {i+1}")

plt.tight_layout()
plt.savefig("output/ps5-2-1-e.png")


wtrain_t = U.T@A

W_Training = wtrain_t.T
print("W_training dimensions", W_Training.shape)

classes = np.arange(1, 41)
y_train = np.repeat(classes, 8)

T_test = np.zeros((10304, 80))
test_images = [f for f in os.listdir(test_dir) if f.endswith('.pgm')]
for i, img_name in enumerate(test_images):
    img_path = os.path.join(test_dir, img_name)
    img_array = np.array(Image.open(img_path))
    T_test[:, i] = img_array.ravel()

A_test = T_test - m
wtest_t = U.T@A_test
W_Testing = wtest_t.T
y_test = np.repeat(classes, 2)
print("W_testing dimensions: ", W_Testing.shape)


# 2.3 Face Recognition: 

# 2.3 a) 

K_vals = [1, 3, 5, 7, 9, 11]
accuracy = []
for k in K_vals:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(W_Training, y_train)
    y_pred = knn.predict(W_Testing)
    accuracy.append(accuracy_score(y_test, y_pred))

print('2.3 outputs: ')
print("K Values:", K_vals)
print("Accuracy: ", accuracy)


# 2.3 b)
lin_ova = SVC(kernel='linear')
lin_ovo = OneVsOneClassifier(lin_ova)

poly_ova = SVC(kernel='poly', degree=3)
poly_ovo = OneVsOneClassifier(poly_ova)

rbf_ova = SVC(kernel='rbf')
rbf_ovo = OneVsOneClassifier(rbf_ova)

# Linear OVA
start_time = time.time()
lin_ova.fit(W_Training, y_train)
y_pred = lin_ova.predict(W_Testing)
lin_ova_acc = accuracy_score(y_test, y_pred)
lin_ova_time = time.time() - start_time

# Linear OVO
start_time = time.time()
lin_ovo.fit(W_Training, y_train)
y_pred = lin_ovo.predict(W_Testing)
lin_ovo_acc = accuracy_score(y_test, y_pred)
lin_ovo_time = time.time() - start_time

# Polynomial OVA
start_time = time.time()
poly_ova.fit(W_Training, y_train)
y_pred = poly_ova.predict(W_Testing)
poly_ova_acc = accuracy_score(y_test, y_pred)
poly_ova_time = time.time() - start_time

# Polynomial OVO
start_time = time.time()
poly_ovo.fit(W_Training, y_train)
y_pred = poly_ovo.predict(W_Testing)
poly_ovo_acc = accuracy_score(y_test, y_pred)
poly_ovo_time = time.time() - start_time

# RBF OVA
start_time = time.time()
rbf_ova.fit(W_Training, y_train)
y_pred = rbf_ova.predict(W_Testing)
rbf_ova_acc = accuracy_score(y_test, y_pred)
rbf_ova_time = time.time() - start_time

# RBF OVO
start_time = time.time()
rbf_ovo.fit(W_Training, y_train)
y_pred = rbf_ovo.predict(W_Testing)
rbf_ovo_acc = accuracy_score(y_test, y_pred)
rbf_ovo_time = time.time() - start_time

print("Linear OVA Training Time:", lin_ova_time)
print("Linear OVO Training Time:", lin_ovo_time)
print("Polynomial OVA Training Time:", poly_ova_time)
print("Polynomial OVO Training Time:", poly_ovo_time)
print("RBF OVA Training Time:", rbf_ova_time)
print("RBF OVO Training Time:", rbf_ovo_time)



print("Linear OVA Accuracy:", lin_ova_acc)
print("Linear OVO Accuracy:", lin_ovo_acc)
print("Polynomial OVA Accuracy:", poly_ova_acc)
print("Polynomial OVO Accuracy:", poly_ovo_acc)
print("RBF OVA Accuracy:", rbf_ova_acc)
print("RBF OVO Accuracy:", rbf_ovo_acc)