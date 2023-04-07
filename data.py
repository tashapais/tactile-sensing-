import numpy as np
from argparse import ArgumentParser
import numpy as np
from PIL import Image
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

data = np.load('train_data_batch_1.npz', allow_pickle=True)

labels, data, mean = data['labels'], data['data'], data['mean']

print('Labels')
print(labels)

print('data')
print(data)

print('mean')
print(mean)

#8 x 8 x 3 = 192 different pixels that we can work with 
print("Data")
print(type(data))
print(data.ndim)
print(data.shape)

print(type(labels))
print(len(labels))
print(labels.ndim)
print(labels.shape)

print(type(mean))
print(mean.ndim)
print(mean.shape)

data = data/np.float32(255)
mean = mean/np.float32(255)
data -= mean
labels = [label-1 for label in labels]
print(labels)