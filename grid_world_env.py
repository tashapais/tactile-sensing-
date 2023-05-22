import gym 
import numpy as np 
import misc_utils as mu 
import pybullet_utils as pu
from math import radians
from gym.utils import seeding
import matplotlib.pyplot as plt
import time
from data import DataLoader
import torchvision
import torch 
from copy import deepcopy

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
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0]), high=np.array([1, HEIGHT-1, WIDTH-1]), dtype=np.uint8)
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
          
    def reset(self):
        """ return initial observations"""
        # red is unexplored in visualization
        self.img_visualization = np.full((HEIGHT, WIDTH, 3), [255, 0, 0], dtype=np.uint8)
        self.img_gt = self.image

        self.img_gt = torch.mul(self.img_gt,0.5) + 0.5
        self.img_gt = self.img_gt.numpy()
        self.img_gt = np.transpose(self.img_gt, (1, 2, 0))
        self.img_gt =  np.multiply(self.img_gt,255)

        initial_loc = np.random.randint(low=(0, 0), high=(HEIGHT, WIDTH))
        
        self.current_step = 0
        self.renderer = plt.imshow(self.img_visualization)
        pixel_value = self.img_gt[tuple(initial_loc)]
        self.img_visualization[tuple(initial_loc)] = np.array([0, 0, 0]) if np.array_equal(pixel_value, np.array([0, 0, 0])) else pixel_value

        ob = (pixel_value, (initial_loc[0], initial_loc[1]))
        self.current_loc = initial_loc
        self.current_step += 1
        return ob

    def step(self, action):
        move = action['move']
        prediction = action['prediction']
        max_prob = action['max_prob']
        probs = action['probs']
        done = action['done']

        new_loc = self.compute_next_loc(move)
        pixel_value = self.img_gt[tuple(new_loc)]
        ob = (pixel_value, (new_loc[0], new_loc[1]))
        self.img_visualization[tuple(new_loc)] = np.array([0, 0, 0]) if np.array_equal(pixel_value, np.array([0, 0, 0])) else pixel_value
        self.current_step += 1
        self.current_loc = new_loc

        reward = 1
        info = {'discover': self.discover,
                'ob': deepcopy(self.img_visualization),
                'label': self.label,
                'prediction':prediction}
        done = self.current_step == self.max_ep_len
        return ob, reward, done, info

    def render(self, mode='human'):
        if mode == 'rgb_array':
            plt.imshow(self.img_visualization)  # return RGB frame suitable for video
            plt.pause(0.00001)
        elif mode == 'human':
            # pop up a window for visualization
            self.renderer.set_data(self.img_visualization)
            plt.pause(0.00001)
            return self.img_visualization
        else:
            super(GridWorldEnv, self).render(mode=mode)  # just raise an exception

    def compute_next_loc(self, action):
        # clockwise action
        if action == 0:
            # move up
            x = self.current_loc[0] if self.current_loc[0] == 0 else self.current_loc[0] - 1
            y = self.current_loc[1]
        elif action == 1:
            # move right
            x = self.current_loc[0]
            y = self.current_loc[1] if self.current_loc[1] == WIDTH-1 else self.current_loc[1] + 1
        elif action == 2:
            # move down
            x = self.current_loc[0] if self.current_loc[0] == HEIGHT-1  else self.current_loc[0] + 1
            y = self.current_loc[1]
        elif action == 3:
            # move left
            x = self.current_loc[0]
            y = self.current_loc[1] if self.current_loc[1] == 0 else self.current_loc[1] - 1
        else:
            raise NotImplementedError('no such action!')
        return (x, y)

if __name__ == "__main__":
    data_loader = DataLoader(batch_size=NUM_EPISODES)
    train_loader = data_loader.return_trainloader()
    test_loader = data_loader.return_testloader()
    train_iter = iter(train_loader)
    test_iter = iter(test_loader)
    images, labels = next(train_iter)
    
    for i in range(NUM_EPISODES):
        label, image = labels[i], images[i]
        grid_world_env = GridWorldEnv(max_ep_len=MAX_EP_LEN, 
                                label=label,
                                image=image)
        initial_ob = grid_world_env.reset()
        #grid_world_env.render(mode='rgb_array')
        done = False
        while not done:
            action = grid_world_env.action_space.sample()
            #grid_world_env.render()
            obs, reward, done, info = grid_world_env.step(action)
            if done:
                print(grid_world_env.current_step)
            #grid_world_env.render()
        print(i)