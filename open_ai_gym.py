import gym 
import numpy as np 
import misc_utils as mu 
import pybullet_utils as pu
from math import radians
from gym.utils import seeding
import matplotlib.pyplot as plt


NUM_CLASSES = 1000
action_map = {
    0: 'up',
    1: 'left',
    2: 'down',
    3: 'right'
}


class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, max_ep_len):
        # clockwise, 0 -> up, 1 -> left, 2 -> down, 3 -> right
        self.action_space = gym.spaces.Discrete(4)
        # (pixel value, axis 0 index, axis 1 index)
        # observation space is left and right inclusive
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0]), high=np.array([1, 8, 8]), dtype=np.uint8)
        self.max_ep_len = max_ep_len

        self.current_loc = None
        self.current_step = 0
        self.images = np.load('datasets/tiny/grids.npy')
        self.img_gt = None
        self.img_index = None
        self.img_visualization = None
        self.renderer = None
          
    def reset(self):
        """ return initial observations"""
        # red is unexplored in visualization
        self.img_visualization = np.full((8, 8, 3), [255, 0, 0], dtype=np.uint8)
        self.img_index = np.random.randint(low=0, high=10)
        self.img_gt = self.images[self.img_index]
        initial_loc = np.random.randint(low=(0, 0), high=(28, 28))
        self.current_step = 0
        self.renderer = plt.imshow(self.img_visualization)

        pixel_value = self.img_gt[tuple(initial_loc)]
        self.img_visualization[tuple(initial_loc)] = np.array([0, 0, 0]) if pixel_value == 1 else np.array([255, 255,
                                                                                                            255])
        ob = np.array([pixel_value, initial_loc[0], initial_loc[1]])
        self.current_loc = initial_loc
        self.current_step += 1
        return ob

    def step(self, action):
        new_loc = self.compute_next_loc(action)
        pixel_value = self.img_gt[tuple(new_loc)]
        ob = np.array([pixel_value, new_loc[0], new_loc[1]])
        self.img_visualization[tuple(new_loc)] = np.array([0, 0, 0]) if pixel_value == 1 else np.array([255, 255, 255])
        self.current_step += 1
        self.current_loc = new_loc

        reward = 1
        info = {}
        done = self.current_step == self.max_ep_len
        return ob, reward, done, info

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.img_visualization  # return RGB frame suitable for video
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
            y = self.current_loc[1] if self.current_loc[1] == 27 else self.current_loc[1] + 1
        elif action == 2:
            # move down
            x = self.current_loc[0] if self.current_loc[0] == 27 else self.current_loc[0] + 1
            y = self.current_loc[1]
        elif action == 3:
            # move left
            x = self.current_loc[0]
            y = self.current_loc[1] if self.current_loc[1] == 0 else self.current_loc[1] - 1
        else:
            raise NotImplementedError('no such action!')
        return (x, y)
    

if __name__ == "__main__":
    num_episodes = 500
    max_ep_len = 1000 

    grid_world_env = GridWorldEnv(max_ep_len=max_ep_len)
    
    for _ in range(num_episodes):
        initial_ob = grid_world_env.reset()
        grid_world_env.render()
        done = False

        while not done:
            action = grid_world_env.action_space.sample()
            obs, reward, done, info = grid_world_env.step(action)

            if done:
                print(grid_world_env.current_step)
            
            grid_world_env.render()




