import random
import time
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

# Use smaller sample for time
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

def time_function(f, *args):
    """ Call a function and return how long it took """
    tic = time.time()
    ret = f(*args)
    toc = time.time()
    return toc-tic, ret

def predict(dists):
    """ Predict labels and determine accuracy """
    y_test_pred = clf.predict_labels(dists, k=1)
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / y_test.shape[0]
    print(f'Correct: {num_correct}/{y_test.shape[0]} -> Accuracy: {accuracy}')

# Test classifier using distance computation with 2 loops
two_loop_time, dists = time_function(clf.compute_distances_two_loops, X_test)
print('Distance computed shape: ', dists.shape)
print('\n')
#plt.imshow(dists, interpolation='none')
#plt.show()

# Get distance using partial vectorization with 1 loop and compare with 2
one_loop_time, dists_one = time_function(clf.compute_distances_one_loop, X_test)
difference = np.linalg.norm(dists - dists_one, ord='fro')
print(f"Difference one loop: {difference}")

# Get distance using full vectorization and compare with no vectorization
no_loop_time, dists_two = time_function(clf.compute_distances_no_loops, X_test)
difference = np.linalg.norm(dists - dists_two, ord='fro')
print(f"Difference two loops: {difference}")

print(f"Times:\nTwo loop: {two_loop_time}\nOne loop: {one_loop_time}\nNo loop: {no_loop_time}\n")

