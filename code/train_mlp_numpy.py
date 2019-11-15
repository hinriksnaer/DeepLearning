"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  assert len(predictions) == len(targets)

  correct = 0
  for i in range(len(predictions)):
    
    if np.argmax(predictions[i])== np.argmax(targets[i]):
      correct += 1
  
  accuracy = correct/len(predictions)

  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Get negative slope parameter for LeakyReLU
  neg_slope = FLAGS.neg_slope

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  """
  Initialize data module
  """
  cifar10=cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)
  x, y = cifar10['train'].next_batch(1)
  x_test, y_test = cifar10['test'].next_batch(10000)
  x = x.reshape(x.shape[0], -1)
  x_test = x_test.reshape(x_test.shape[0], -1)
  """
  initialize the network
  """
  network = MLP(x.shape[1], dnn_hidden_units, y.shape[1], neg_slope)

  
  """
  compute forward
  """
  
  crossEntropy = CrossEntropyModule()

  """
  batch gradient descent
  """
  for i in range(FLAGS.max_steps):
    x, y = cifar10['train'].next_batch(FLAGS.batch_size)
    x = x.reshape(x.shape[0], -1)
    prediction = network.forward(x)
    loss = crossEntropy.forward(prediction, y)

    gloss = crossEntropy.backward(prediction, y)
    network.backward(gloss)

    for linearModule in network.linearModules:
      linearModule.params['weight'] = linearModule.params['weight'] - FLAGS.learning_rate * 1/FLAGS.batch_size * linearModule.grads['weight']
      linearModule.params['bias'] = linearModule.params['bias'] - FLAGS.learning_rate * 1/FLAGS.batch_size * linearModule.grads['bias']

    if i%FLAGS.eval_freq == 0:
      prediction = network.forward(x_test)

      print('Accuracy after '+ str(i) +' steps ' + str(accuracy(prediction, y_test)))
  prediction = network.forward(x_test)
  print('Final accuracy')
  print(accuracy(prediction, y_test))
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
                      help='Negative slope parameter for LeakyReLU')
  FLAGS, unparsed = parser.parse_known_args()

  main()