from __future__ import print_function
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers.linear_svm import svm_loss_naive
from cs231n.classifiers.linear_svm import svm_loss_vectorized
from cs231n.gradient_check import grad_check_sparse

# Load raw data
cifar10_dir = '../datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# Print size of training and test data
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
print('\n')

# Split data into train, val, test, and dev sets
num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500
# Validation set
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]
# Train set
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]
# Dev set (subset of training set for quick use)
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]
y_dev = y_train[mask]
# Test set - use num_test test points
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]
# Check
print('RESIZE:\n')
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
print('\n')

# Preprocess: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1)) # second dim becomes 32x32x3=3072
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_dev= np.reshape(X_dev, (X_dev.shape[0], -1))
# Preprocess: compute and subtract the mean image from the training/test data
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
X_dev -= mean_image
# Append bias dimension of ones (bias trick) so SVM only needs to optimize
# a single weight matrix W
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))]) # second dim becomes 3072+1=3073
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

''' Evaluate naive implementation of loss '''
# Generate a random SVM weight matrix of small numbers
W = np.random.randn(3073, 10) * 0.0001 # 3073x10 (10 classes)
loss, grad = svm_loss_naive(W, X_dev, y_dev, 0)

# # Numerically compute the gradient along several randomly chosen dimensions
# #   and compare with analytically computed gradient (grad)
# f = lambda w: svm_loss_naive(w, X_dev, y_dev, 0.0)[0] # Returns the loss
# grad_numerical = grad_check_sparse(f, W, grad)
# # Again with the regularization turned on
# loss, grad = svm_loss_naive(W, X_dev, y_dev, 5e1)
# f = lambda w: svm_loss_naive(w, X_dev, y_dev, 5e1)[0] # Returns the loss
# grad_numerical = grad_check_sparse(f, W, grad)

''' Evaluate vectorized implementation of loss '''
loss_v, grad_v = svm_loss_vectorized(W, X_dev, y_dev, 0)
print(np.linalg.norm(grad - grad_v))
print(loss - loss_v)