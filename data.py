import torch 
import torchvision
import torchvision.transforms as transforms

class CIFARDataLoader():
    def __init__(self, batch_size):
        self.training_data = None 
        self.training_data_labels = None 
        self.classes = ('plane', 
                        'car', 
                        'bird', 
                        'cat',
                        'deer', 
                        'dog',
                        'frog', 
                        'horse', 
                        'ship', 
                        'truck')
        self.batch_size = batch_size
        self.transform = transforms.Compose(
    [transforms.ToTensor()])
                        

    def return_trainloader(self):
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                                download=True, transform=self.transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                shuffle=True, num_workers=0)

        return trainloader


    def return_testloader(self):
        testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=self.transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                         shuffle=False, num_workers=0)
        
        return testloader