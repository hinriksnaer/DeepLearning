"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import torch
from torch import nn

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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
  predictions = predictions.cpu().detach().numpy()
  targets = targets.cpu().detach().numpy()
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
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """
  if torch.cuda.is_available():  
    dev = "cuda:0" 
  else:  
    dev = "cpu"  
  device = torch.device(dev) 
  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  """
  Initialize data module
  """
  cifar10=cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)
  x, y = cifar10['train'].next_batch(1)
  x_test, y_test = cifar10['test'].next_batch(10000)

  x_test = torch.tensor(x_test)
  y_test = torch.tensor(y_test)

  """
  initialize the network
  """
  network = ConvNet(x.shape[1], y.shape[1]).to(device)
  crossEntropy = nn.CrossEntropyLoss()
  
  optimizer = None

  optimizer = torch.optim.Adam(network.parameters(), lr=FLAGS.learning_rate, amsgrad=True)
  store_loss = None
  for i in range(FLAGS.max_steps):
    x, y = cifar10['train'].next_batch(FLAGS.batch_size)
    x = torch.tensor(x).to(device)
    y = torch.LongTensor(y).to(device)
    prediction = network.forward(x)

    loss = crossEntropy.forward(prediction, torch.max(y, 1)[1])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    store_loss = loss.cpu()
    del loss
    del x
    del y
    del prediction
    prediction_collection = None
    if i%FLAGS.eval_freq == 0:
      with torch.no_grad():
        print('Loss after '+str(i)+' steps '+str(store_loss))
        for j in range(100):
          test_data = x_test[j*100:j*100+100].to(device)
          prediction = network.forward(test_data)
          prediction = nn.functional.softmax(prediction)
          del test_data
          if j == 0:
            prediction_collection = prediction
          else:
            prediction_collection = torch.cat((prediction_collection, prediction), 0)
          del prediction
        print('Accuracy after '+ str(i) +' steps ' + str(accuracy(prediction_collection, y_test)))

  prediction_collection = None
  with torch.no_grad():
    print('final Loss',store_loss)
    for j in range(100):
      test_data = x_test[j*100:j*100+100].to(device)
      prediction = network.forward(test_data).cpu()
      prediction = nn.functional.softmax(prediction)
      if j == 0:
        prediction_collection = prediction
      else:
        prediction_collection = torch.cat((prediction_collection, prediction), 0)
      del prediction
    print('Final accuracy')
    print(accuracy(prediction_collection, y_test))


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
  FLAGS, unparsed = parser.parse_known_args()

  main()