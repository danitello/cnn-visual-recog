import random
import numpy
import matplotlib as plt
from cs231n.data_utils import load_CIFAR10

# Delete possibly previously loaded data to avoid memory issue
try:
    del X_train, y_train, X_test, y_test
except:
    pass

# Load raw CIFAR-10 data
cifar10_dir = '../datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# Print size of training and test data
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
