
import scipy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow import keras
from keras import models, layers
from keras.preprocessing import image
from os import listdir
from PIL import Image
from enum import Enum
import cv2


# class Vehicle(Enum):
#     airplane = 1
#     automobile = 2
#     truck = 3

# # 0: Data Preprossesing
# data2 = scipy.io.loadmat('input/HW6_Data2_full.mat')
# X = np.array(data2['X'])
# y = np.array(data2['y_labels'])
# num_classes = 3

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(2/15))
# X_train = X_train.reshape((13000, 32, 32, 1))

# indices = random.sample(range(X_train.shape[0]), 12)
# fig, axes = plt.subplots(3, 4, figsize=(8, 6))
# for ax, i in zip(axes.flat, indices):
#     img = X_train[i].squeeze()
#     ax.imshow(img, cmap='gray')
#     lbl = int(np.squeeze(y_train[i]))
#     try:
#         ax.set_title(Vehicle(lbl).name)
#     except:
#         ax.set_title(str(lbl))
#     ax.axis('off')
# plt.tight_layout()
# plt.savefig('output/ps7-1-b-1.png')

# X_test = X_test.reshape((2000, 32, 32, 1))
# indices = random.sample(range(X_test.shape[0]), 12)
# fig, axes = plt.subplots(3, 4, figsize=(8, 6))
# for ax, i in zip(axes.flat, indices):
#     img = X_test[i].squeeze()
#     ax.imshow(img, cmap='gray')
#     lbl = int(np.squeeze(y_train[i]))
#     try:
#         ax.set_title(Vehicle(lbl).name)
#     except:
#         ax.set_title(str(lbl))
#     ax.axis('off')
# plt.tight_layout()
# plt.savefig('output/ps7-1-c-1.png')


# y_train_vec = []

# y_train_flat = y_train.flatten() - 1  # Flatten and adjust to 0-indexed
# y_train_vec = np.eye(num_classes)[y_train_flat]

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(32, (5, 5), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(num_classes, activation='softmax'))

# model.summary()

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# history = model.fit(np.asarray(X_train), np.asarray(y_train_vec), epochs=150, batch_size=15)


# p71e2 = plt.figure()
# plt.plot(history.history['accuracy'], label='accuracy')
# #plt.plot(history.history['val_accuracy'], label = 'val_accuracy') # validation accuracy; no validation in this example
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# p71e2.savefig('output/ps7-1-e-2.png')



# preds = model.predict(X_test)
# pred_labels = np.argmax(preds, axis=1) + 1
# y_test_flat = y_test.flatten()
# accuracy = accuracy_score(y_test_flat, pred_labels)
# print(f'Test Accuracy: {accuracy}')


def load_data(img_dir, num_classes):
    imgs = []
    labels = []
    for img_file in listdir(img_dir):
        img = preprocess(img_dir, img_file)
        imgs.append(img)
        label = int(img_file.split('_')[0])
        labels.append(label)
    return imgs, labels

# Preprocess images
def preprocess(img_dir, img_file):
    img = image.img_to_array(image.load_img(img_dir + '/' + img_file))
    img = img.astype(np.float64) - np.mean(img)
    img /= np.std(img)
    return img

# Initialize the model
num_classes = 2
imgs, labels = load_data('./input/p2/train_imgs', num_classes)
print(labels[0:10])


model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=imgs[0].shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(10, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
#model.add(layers.Dense(num_hidden_units, activation = 'sigmoid'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

# Compile the model
model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(np.asarray(imgs), np.asarray(labels), epochs=4)

# Print the accuracy from the final training epoch
acc_list = history.history.get('accuracy') or history.history.get('acc')
if acc_list:
    print(f'accuracy: {acc_list[-1]:.4f}')




def sliding_window(img, step_size, window_size):
    """Generator that yields (x, y, window) for each step."""
    for y in range(0, img.shape[0] - window_size[1], step_size):
        for x in range(0, img.shape[1] - window_size[0], step_size):
            yield (x, y, img[y:y + window_size[1], x:x + window_size[0], :])

def preprocess_window(window):
    """Preprocess to match training conditions."""
    window = window.astype(np.float64)
    window -= np.mean(window)
    window /= np.std(window) + 1e-8
    return window

def detect_object(model, img, window_size=(64, 64), step_size=32, threshold=0.8):
    """Find only the highest-confidence detection above threshold."""
    H, W = img.shape[:2]
    h, w = window_size

    # Calculate how many windows total
    num_x = (W - w) // step_size + 1
    num_y = (H - h) // step_size + 1
    total_windows = num_x * num_y
    
    best_box = None
    best_score = -1.0
    count = 0
    # Slide window
    for (x, y, window) in sliding_window(img, step_size, window_size):
        if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
            continue  # skip incomplete windows
        
        patch = preprocess_window(window)
        patch = np.expand_dims(patch, axis=0)
        pred = model.predict(patch, verbose=0)[0][0]
        print(f"Window ({count}/{total_windows}) Prediction: {pred:.4f}")
        if pred > best_score:
            best_score = pred
            best_box = (x, y, window_size[0], window_size[1])
        count += 1
    # Draw only if score exceeds threshold
    img_copy = img.copy().astype(np.uint8)
    if best_score >= threshold and best_box is not None:
        x, y, w, h = best_box
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_copy, f'{best_score:.2f}', (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return img_copy, best_score

test_images, test_labels = load_data('./input/p2/test_imgs', 2)

K = 1
for img in test_images:
    print("Processing Image ", K)
    img, score = detect_object(model, img, window_size=(32, 96), step_size=4, threshold=0.7)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.axis('off')
    plt.imshow(rgb_img)
    plt.savefig(f'output/ps7-2-c-{K}.png')
    K += 1