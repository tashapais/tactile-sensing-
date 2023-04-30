import torch 
import torch.nn as nn 
import torch.nn.functional as F


"""The discriminator works on a partially revealed image to understand where to go next. 
"""



class Discriminator_NN(nn.Module):
    def __init__(self, height, width):
        super(Discriminator_NN, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        # Max pooling over a (2, 2) window 
        # If the size is a square, you can specify with a single number 

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x 

