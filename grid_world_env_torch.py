import gym 
import torch
import misc_utils as mu 
from math import radians
from gym.utils import seeding
import matplotlib.pyplot as plt
import time
from data import DataLoader
import torchvision
from copy import deepcopy
import numpy as np

action_map = {
    0: 'up',
    1: 'left',
    2: 'down',
    3: 'right'
}
HEIGHT, WIDTH = 32, 32
NUM_EPISODES = 1000
MAX_EP_LEN = 5000

class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, max_ep_len, label, image):
        self.action_space = gym.spaces.Discrete(4)
        self.max_ep_len = max_ep_len
        self.current_loc = None
        self.current_step = 0
        self.label, self.image = label, image
        self.img_gt = None
        self.img_index = None
        self.img_visualization = None
        self.renderer = None
        self.discover = True
        self.explorer = None 
        self.discriminator = None 
        self.move_dim = 4
        self.num_classes = 10
        self.action_space = gym.spaces.Dict({"move": gym.spaces.Discrete(self.move_dim),
                                             "prediction": gym.spaces.Discrete(self.num_classes),
                                             "probs": gym.spaces.Box(low=0, high=1, shape=(self.num_classes,)),
                                             "max_prob": gym.spaces.Box(low=0, high=1, shape=(1,)),
                                             "done": gym.spaces.Discrete(2)})
        self.observation_space = gym.spaces.Box(low=np.zeros((3, HEIGHT, WIDTH)),
                                                high=np.full((3, HEIGHT, WIDTH), 255), dtype=np.uint8)
        
    def reset(self):
        """ return initial observations"""
        # red is unexplored in visualization
        self.img_visualization = torch.full((3, HEIGHT, WIDTH), 255, dtype=torch.uint8)
        self.img_gt = self.image
        initial_loc = torch.randint(low=0, high=32, size=(2,))
        self.current_step = 0
        self.renderer = plt.imshow (self.img_visualization.permute(1,2,0).numpy())
        pixel_value = self.img_gt[:,initial_loc[0],initial_loc[1]]
        self.img_visualization[:,initial_loc[0],initial_loc[1]] = torch.tensor([0, 0, 0]) if torch.equal(pixel_value, torch.tensor([0, 0, 0])) else pixel_value

        self.current_loc = initial_loc
        self.current_step += 1
        return self.img_visualization
    
    def done(self):
        return self.current_step>=self.max_ep_len

    def step(self, action):
        if type(action) == torch.Tensor:
            done = self.current_step ==  self.max_ep_len
            new_loc = self.compute_next_loc(action)
            pixel_value = self.img_gt[:,new_loc[0], new_loc[1]]
            discover = not torch.equal(pixel_value, self.img_visualization[:,new_loc[0],new_loc[1]])
            self.img_visualization[:,new_loc[0], new_loc[1]] =  pixel_value
            ob = self.img_visualization
            self.current_step += 1
            self.current_loc = new_loc
            return done, ob
        
        else:
            move = action['move']
            prediction = action['prediction']
            done = action['done']

            new_loc = self.compute_next_loc(move)
            pixel_value = self.img_gt[:,new_loc[0],new_loc[1]]
            discover = not torch.equal(pixel_value, self.img_visualization[:,new_loc[0],new_loc[1]])
            self.img_visualization[:,new_loc[0],new_loc[1]] =  pixel_value
            ob = self.img_visualization
            self.current_step += 1
            self.current_loc = new_loc
            reward = 1 if prediction == self.label else 0

            done = self.current_step == self.max_ep_len

            info = {'discover': discover,
                    'img': deepcopy(ob),
                    'label': self.label,
                    'prediction':prediction}
            
            done = self.current_step == self.max_ep_len
            return ob, reward, done, info



    def compute_next_loc(self, action):
        if action == 0:
            x = self.current_loc[0] if self.current_loc[0] == 0 else self.current_loc[0] - 1
            y = self.current_loc[1]
        elif action == 1:
            x = self.current_loc[0]
            y = self.current_loc[1] if self.current_loc[1] == WIDTH-1 else self.current_loc[1] + 1
        elif action == 2:
            x = self.current_loc[0] if self.current_loc[0] == HEIGHT-1  else self.current_loc[0] + 1
            y = self.current_loc[1]
        elif action == 3:
            x = self.current_loc[0]
            y = self.current_loc[1] if self.current_loc[1] == 0 else self.current_loc[1] - 1
        else:
            raise NotImplementedError('no such action!')
        return (x, y)