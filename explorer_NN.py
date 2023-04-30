import torch 
import torch.nn as nn 



'''Explorer takes in observations and decides where to move next.'''


class Explorer_NN(nn.Module):

    def __init__(self, height, weight):

        super(Explorer_NN, self).__init__()

        self.