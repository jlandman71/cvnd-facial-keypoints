## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
import torchvision.models as models


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        
        
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2) # input 224x224  output 220x220 
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)  # input 224x224  output 220x220 
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2) 
        self.conv4 = nn.Conv2d(128, 256, 5, padding=2) 
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(256*14*14, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        
        # dropout with p=0.3
        self.fc_drop = nn.Dropout(p=0.3)
        
        # 136 output channels
        self.fc3 = nn.Linear(1000, 136)

                
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        x = self.pool(F.relu(self.conv1(x))) # input 224x224  output 112x112
        x = self.pool(F.relu(self.conv2(x))) # input 112x112 output 56x56
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv3(x))) # input 56x56 output 28x28
        x = self.bn2(x)
        x = self.pool(F.relu(self.conv4(x))) # input 28x28 output 14x14
        x = self.bn3(x)
        
        # prep for linear layer
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc_drop(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x