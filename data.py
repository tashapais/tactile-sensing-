import numpy as np
from argparse import ArgumentParser
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
import torch


import matplotlib.pyplot as plt

data = np.load('train_data_batch_1.npz', allow_pickle=True)

labels, data, mean = data['labels'], data['data'], data['mean']





data = data/np.float32(255)
mean = mean/np.float32(255)
data -= mean
labels = [label-1 for label in labels]


for img in data:
    img = img.reshape((8, 8, 3))
    plt.imshow(img)
    plt.show()
    break

#we have a thousand labels for this downsized version of Imagenet
#print(labels)
#print(len(set(labels)))
## we have the data now let's start building the open ai gym environment