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
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                        

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
    

# print("1")
# data_loader = DataLoader(batch_size=1)
# print("2")
# train_loader = data_loader.return_trainloader()
# # test_loader = data_loader.return_testloader()
# for i, data in enumerate(train_loader, 0):
#     # Get the inputs and labels
#     images, labels = data
#     # Print batch number and shape of input tensor
#     print("The data shape of the image below")
#     print(images.shape)
#     print("The shape of the labels")
#     print(labels.shape)
#     print("The first image we have:")
#     print(images[0], images[0].shape)
#     print("The first label we have")
#     print(labels[0], labels[0].shape)
#     print("The labels and the image above are listed")
#     print("Here are the images above")
#     images = torch.transpose(images, 1, 2)
#     images = torch.transpose(images, 2, 3)
#     print("The data shape of the image below")
#     print(images.shape)
#     plt.imshow(torchvision.utils.make_grid(images))
#     time.sleep(5)
#     if i == 10:
#         break 
# test_iter = iter(test_loader)
#image, label = next(train_iter)
# print(image,type(image))
# print(label, type(label))