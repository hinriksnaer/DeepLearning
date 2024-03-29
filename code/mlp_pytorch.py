"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes, neg_slope):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
      neg_slope: negative slope parameter for LeakyReLU

    TODO:
    Implement initialization of the network.
    """
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(MLP, self).__init__()

    network = []

    first_output_length = n_classes if len(n_hidden) == 0 else int(n_hidden[0])

    network.append(nn.Linear(n_inputs, first_output_length))
    

    for i in range(len(n_hidden)):
      last = i == len(n_hidden) - 1
      
      n_outputs = n_classes if last else n_hidden[i+1]

      network.append(nn.LeakyReLU(neg_slope))
      network.append(nn.Linear(n_hidden[i], n_outputs))

    self.module_list = nn.ModuleList(network)

    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    for module in self.module_list:
      x = module.forward(x)
    out = x
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
