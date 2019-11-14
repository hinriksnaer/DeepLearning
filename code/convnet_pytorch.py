"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
import torch.nn.functional as F

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    super(ConvNet, self).__init__()

    self.conv1 = nn.Conv2d(
      in_channels=n_channels, 
      out_channels=64, 
      kernel_size = 3, 
      stride=1, 
      padding=1,
      )

    self.batchnorm1 = nn.BatchNorm2d(
      num_features=64,
      )

    self.maxpool1 = nn.MaxPool2d(
      kernel_size=3,
      stride=2, 
      padding=1,
      )
    
    self.conv2 = nn.Conv2d(
      in_channels=64,
      out_channels=128,
      kernel_size=3,
      padding=1,
      stride=1,
    )

    self.batchnorm2 = nn.BatchNorm2d(
      num_features=128,
      )

    self.maxpool2 = nn.MaxPool2d(
      kernel_size=3,
      stride=2,
      padding=1,
    )

    self.conv3a = nn.Conv2d(
      in_channels=128,
      out_channels=256,
      kernel_size=3,
      stride=1,
      padding=1,
    )

    self.batchnorm3 = nn.BatchNorm2d(
      num_features=256,
      )

    self.conv3b = nn.Conv2d(
      in_channels=256,
      out_channels=256,
      kernel_size=3,
      stride=1,
      padding=1,
    )

    self.batchnorm4 = nn.BatchNorm2d(
      num_features=256,
      )

    self.maxpool3 = nn.MaxPool2d(
      kernel_size=3,
      stride=2,
      padding=1,
    )

    self.conv4a = nn.Conv2d(
      in_channels=256,
      out_channels=512,
      kernel_size=3,
      stride=1,
      padding=1,
    )

    self.batchnorm5 = nn.BatchNorm2d(
      num_features=512,
      )

    self.conv4b = nn.Conv2d(
      in_channels=512,
      out_channels=512,
      kernel_size=3,
      stride=1,
      padding=1,
    )

    self.batchnorm6 = nn.BatchNorm2d(
      num_features=512,
      )

    self.maxpool4 = nn.MaxPool2d(
      kernel_size=3,
      stride=2,
      padding=1,
    )

    self.conv5a = nn.Conv2d(
      in_channels=512,
      out_channels=512,
      kernel_size=3,
      stride=1,
      padding=1,
    )

    self.batchnorm7 = nn.BatchNorm2d(
      num_features=512,
      )

    self.conv5b = nn.Conv2d(
      in_channels=512,
      out_channels=512,
      kernel_size=3,
      stride=1,
      padding=1,
    )

    self.batchnorm8 = nn.BatchNorm2d(
      num_features=512,
      )

    self.maxpool5 = nn.MaxPool2d(
      kernel_size=3,
      stride=2,
      padding=1,
    )

    self.avgpool1 = nn.AvgPool2d(
      kernel_size=1,
      stride=1,
      padding=0
    )

    self.linear1 = nn.Linear(512, 10)

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
    x = self.conv1.forward(x)
    x = F.relu(self.batchnorm1.forward(x))
    x = self.maxpool1.forward(x)
    x = self.conv2.forward(x)
    x = F.relu(self.batchnorm2.forward(x))
    x = self.maxpool2.forward(x)
    x = self.conv3a.forward(x)
    x = F.relu(self.batchnorm3.forward(x))
    x = self.conv3b.forward(x)
    x = F.relu(self.batchnorm4.forward(x))
    x = self.maxpool3.forward(x)
    x = self.conv4a.forward(x)
    x = F.relu(self.batchnorm5.forward(x))
    x = self.conv4b.forward(x)
    x = F.relu(self.batchnorm6.forward(x))
    x = self.maxpool4.forward(x)
    x = self.conv5a.forward(x)
    x = F.relu(self.batchnorm7.forward(x))
    x = self.conv5b.forward(x)
    x = F.relu(self.batchnorm8.forward(x))
    x = self.maxpool5.forward(x)
    x = self.avgpool1.forward(x)
    x = x.reshape(x.shape[0],x.shape[1])
    out = self.linear1(x)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
