import numpy as np 
import matplotlib.pyplot as plt
from cs231n.classifiers.neural_net import TwoLayerNet
from cs231n.gradient_check import eval_numerical_gradient

INPUT_SIZE = 4
HIDDEN_SIZE = 10
NUM_CLASSES = 3
NUM_INPUTS = 5

def init_toy_model():
  """ generates test model """
  np.random.seed(0)
  return TwoLayerNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, std=1e-1)

def init_toy_data():
  """ generates test data """
  np.random.seed(1)
  X = 10 * np.random.randn(NUM_INPUTS, INPUT_SIZE)
  y = np.array([0, 1, 2, 2, 1])
  return X, y

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

net = init_toy_model()
X, y = init_toy_data()

# Compute first part of forward pass
scores = net.loss(X)
print('Scores:\n', scores, '\n')
print('correct scores:')
correct_scores = np.asarray([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
print(correct_scores)
print()

# Forward pass scores
print('Score difference: ')
print(np.sum(np.abs(scores - correct_scores)))

# Forward pass loss and backwards pass grads
loss, grads = net.loss(X, y, reg=0.05)
correct_loss = 1.30378789133
print('Loss difference:')
print(np.sum(np.abs(loss - correct_loss)))

# Check implementation of backward pass
for param_name in grads:
  f = lambda W: net.loss(X, y, reg=0.05)[0]
  param_grad_num = eval_numerical_gradient(f, net.params[param_name],verbose=False)
  print(f'{param_name} max relative error: {rel_error(param_grad_num, grads[param_name])}')
