import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import misc_utils as mu
import torch

if __name__ == "__main__":
    # Generating a random image
    #
    # # Using imshow to display the image
    # plt.imshow(image)
    #
    # # Adding a title
    # plt.title('My Image Title')
    #
    # # Displaying the plot
    # plt.show()

    #x = torch.tensor([mu.current_square for _ in range(3)])
    # x = torch.tensor([mu.current_square]*3)
    #
    # print(x)

    t = 0
    while t < 100:
        t += 1
        image = np.random.rand(10, 10)
        plt.cla()
        plt.imshow(image)
        plt.pause(0.001)
        print(t)
