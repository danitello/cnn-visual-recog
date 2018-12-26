import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, y[i]] -= X[i, :] # Subtract example from gradient of correct class
        dW[:, j] += X[i, :] # Add example to gradient of incorrect class

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  # Get the average for loss weight gradient by dividing by the 
  # number of training samples (X) that we used
  dW /= num_train
  # Include the gradient of the regularization 
  dW += reg*W 

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]

  scores = X.dot(W) # num_train x num_classes
  # For every score, subtract
  #   a column of the correct scores
  #   and add 1.
  #
  # This fulfills the sj - syi + 1 formula 
  #   except that the correct score values will be 1 instead of 0
  formula = (scores) - (scores[np.arange(num_train),y].reshape(-1,1)) + (1)
  # Find the max of 0 and the loss value to get the loss of this
  #   training example
  margins = np.maximum(0, formula)
  # Final loss
  loss = np.sum(formula)
  # Average the loss
  loss /= num_train
  # Include regularization loss
  loss += 0.5 * reg * np.sum(W * W)

  # For each incorrect class in W with
  #   loss > 0, we need to add X[i, :] to it (this only needs to be done a max
  #   of 1 time so this case is represented by a 1),
  #   and for each correct class in W, how many times we need to subtract
  #   X[i, :] from it, once for each incorrect class with loss > 0
  #
  #   This is done once for each class, for each image in X, as before
  margins[np.arange(num_train), y] = 0 # Setting margins of correct ones to 0
  margins[margins > 0] = 1
  margins[np.arange(num_train), y] = -np.sum(margins, axis=1)
  dW = X.T.dot(margins)
  # Get average
  dW /= num_train
  # Include regularization loss
  dW += reg*W
  

  return loss, dW
