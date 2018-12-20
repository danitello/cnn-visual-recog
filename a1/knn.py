import random
import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers.k_nearest_neighbor import KNearestNeighbor


""" Use cross validation to determine the best value of the k hyperparam """
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

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

# Split the arrays into individual folds
X_train_folds = np.split(X_train, num_folds)
y_train_folds = np.split(y_train, num_folds)

# Dictionary holding the accuracies (list) for different values of k
k_to_accuracies = {}
# k-fold cross validation using fold i as validation, and all others as training
for choice in k_choices:
    for i in range(num_folds):
        # Partition training and test arrays
        X_tr = np.vstack([X_train_folds[x] for x in range(num_folds) if x!=i])
        y_tr = np.hstack([y_train_folds[x] for x in range(num_folds) if x!=i])
        X_te = X_train_folds[i]
        y_te = y_train_folds[i]
        # Create kNN classifier instance
        clf = KNearestNeighbor()
        clf.train(X_tr, y_tr)
        # Predict
        pred = clf.predict(X_te, k=choice)
        acc = float(np.sum(pred == y_te)) / y_te.shape[0]
        print(f"k = {choice}, accuracy = {acc}")
        if i == 0:
            k_to_accuracies[choice] = [acc]
        else:
            k_to_accuracies[choice].append(acc)

# Plot results
for k in k_choices:
    accs = k_to_accuracies[k]
    plt.scatter([k] * len(accs), accs)
# Plot trend line with error bars corresponding to standard deviation
accs_mean = np.array([np.mean(val) for key,val in sorted(k_to_accuracies.items())])
accs_std = np.array([np.std(val) for key,val in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accs_mean, yerr=accs_std)
plt.title('Cross validation on k')
plt.xlabel('k')
plt.ylabel('Cross validation accuracy')
plt.show()



''' For future reference
def time_function(f, *args):
    """ Call a function and return how long it took """
    tic = time.time()
    ret = f(*args)
    toc = time.time()
    return toc-tic, ret

def two_loop():
    # Test classifier using distance computation with 2 loops 
    two_loop_time, dists = time_function(clf.compute_distances_two_loops, X_test)
    print('Distance computed shape: ', dists.shape)
    print('\n')
    plt.imshow(dists, interpolation='none')
    plt.show()

def one_loop():
    # Get distance using partial vectorization with 1 loop and compare with 2 
    one_loop_time, dists_one = time_function(clf.compute_distances_one_loop, X_test)
    difference = np.linalg.norm(dists - dists_one, ord='fro')
    print(f"Difference one loop: {difference}")

def no_loop():
    # Get distance using full vectorization and compare with no vectorization
    no_loop_time, dists_two = time_function(clf.compute_distances_no_loops, X_test)
    difference = np.linalg.norm(dists - dists_two, ord='fro')
    print(f"Difference two loops: {difference}")
'''