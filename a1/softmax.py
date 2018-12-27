import random
import numpy as np
import matplotlib.pyplot as plt 
from time import time
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers.softmax import softmax_loss_naive
from cs231n.classifiers.softmax import softmax_loss_vectorized
from cs231n.classifiers.linear_classifier import Softmax
from cs231n.gradient_check import grad_check_sparse

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier. 
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = '../datasets/cifar-10-batches-py'
    
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    
    # subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]
    
    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
    
    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image
    
    # add bias dimension and transform into columns
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev

# Get CIFAR10 data
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
print('dev data shape: ', X_dev.shape)
print('dev labels shape: ', y_dev.shape)
print('\n')

''' Generate weight matrix and conduct softmax loss computation using naive version '''
W = np.random.randn(3073, 10) * 1e-4
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)
# Since W is initialized to very small values, loss should come out to ~(-log(0.1))
print(f"naive loss computation: {loss} -log(0.1): {-np.log(0.1)}") 
# Check gradient calculation for accuracy
f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]
grad_check_sparse(f, W, grad, 10)
# Another check, with regularization this time
loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, 0.000005)
f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.000005)[0]
grad_check_sparse(f, W, grad, 10)

''' Repeat for vectorized implementation and compare '''
loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, 0.000005)
print(f'vectorized loss: {loss_vectorized} -log(0.1): {-np.log(0.1)}')

# Use the Frobenius norm to compare the two versions of the gradient.
grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print('Loss difference: ', np.abs(loss_naive - loss_vectorized))
print('Gradient difference: ', grad_difference)

''' Use the validation set to tune hyperparams -
        regularization strength and learning rate '''
results = {} # {(lr,reg) : (acc_tr, acc_val)}
best_acc_val = -1
best_softmax = None
learning_rates = [1e-7, 5e-7]
regularization_strengths = [2.5e4, 5e4]

for i in range(len(learning_rates)):
    for j in range(len(regularization_strengths)):
        sm = Softmax()
        sm.train(X_val, y_val, learning_rate=learning_rates[i],
                reg=regularization_strengths[j], num_iters=1000, verbose=True)
        y_v_pred = sm.predict(X_val)
        y_tr_pred = sm.predict(X_train)
        acc_val = np.mean(y_val == y_v_pred)
        acc_tr = np.mean(y_train == y_tr_pred)
        results[(learning_rates[i], regularization_strengths[j])] = (acc_tr, 
                                                                    acc_val)
        if acc_val > best_acc_val:
            best_acc_val = acc_val
            best_softmax = sm

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print(f"lr {lr}, reg {reg} train accuracy: {train_accuracy} val accuracy: {val_accuracy}")
print("Best validation accuracy achieved during cross-val", best_acc_val)

# Evaluate the best softmax on test set
y_test_pred = best_softmax.predict(X_test)
test_acc = np.mean(y_test == y_test_pred)
print('Final test acc:', test_acc)

''' Visualize the learned weights for each class '''
w = best_softmax.W[:-1,:] # remove bias (ones)
w = w.reshape(32, 32, 3, 10)

w_min, w_max = np.min(w), np.max(w)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)
    
    # Rescale the weights to be between 0 and 255
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])
plt.show()