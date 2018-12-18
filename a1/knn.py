import random
import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers.k_nearest_neighbor import KNearestNeighbor

# Load raw CIFAR-10 data
cifar10_dir = '../datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# Print size of training and test data
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
print('\n')

# User smaller sample for now
X_train = X_train[:5000]
y_train = y_train[:5000]
X_test = X_test[:500]
y_test = y_test[:500]

# Reshape image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print('Training data reshape', X_train.shape)
print('Testing data reshape', X_test.shape)
print('\n')

# Create kNN classifier instance
clf = KNearestNeighbor()
clf.train(X_train, y_train)

# Test classifier
dists = clf.compute_distances_two_loops(X_test)
print('Distance computed shape: ', dists.shape)
print('\n')
#plt.imshow(dists, interpolation='none')
#plt.show()

# Predict
y_test_pred = clf.predict_labels(dists, k=1)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / y_test.shape[0]
print(f'Correct: {num_correct}/{y_test.shape[0]} -> Accuracy: {accuracy}')