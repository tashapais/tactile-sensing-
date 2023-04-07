import time 
import gym 
import numpy as np 
import cv2 
import copy 
import misc_utils as mu 
import pybullet_utils as pu
import math
import os
from math import radians
import itertools
from gym.utils import seeding
import random


NUM_CLASSES = 200

class GridWorldEnv(gym.Env):
    metadata = {'render.modes':['human', 'rgb_array']}
    def __init__(self, 
                 max_ep_len,
                 ob_type,
                 reward_type,
                 action_type,
                 discriminator,
                 tactile_sim,
                 window_id=0):
        
        self.ob_type = ob_type
        self.reward_type = reward_type
        self.action_type = action_type
        self.tactile_sim = tactile_sim
        self.window_id = window_id

        if action_type == 'only_explore':
            # clockwise, 0 -> up, 1 -> left, 2 -> down, 3 -> right
            self.action_space = gym.spaces.Discrete(4)
        elif action_type == 'explore_and_predict':
            # also predict termination and the predicted class
            #predict the 
            self.action_space = gym.spaces.Discrete(4+NUM_CLASSES)

        if ob_type == 'local':
            # (pixel value, height index, width index)
            # observation space is left and right inclusive
            #not sure how this may look now
            self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0]), high=np.array([1, 7, 7]), dtype=np.uint8)
        
        elif ob_type == 'global':
            self.observation_space = gym.spaces.Box(low=np.zeros((1, 8, 8)), high=np.full((1, 8, 8), 255), dtype=np.uint8)
        
        else:
            raise TypeError
        
        self.max_ep_len = max_ep_len
        self.discriminator = discriminator
        self.done_threshold = 1.0
        self.current_loc = None
        self.current_step = 0
        self.images = np.load('datasets/tiny/grids.npy')
        self.img_gt = None
        self.num_gt = None
        self.img_belief = None
        self.rendered = None
        self.discover = None

    def reset(self):
        """ return initial observations"""
        self.discover = True

        #again not sure how to convert from BW to RGB
        self.img_belief = np.full((1, 8, 8), mu.unexplored, dtype=np.uint8)
        self.num_gt = np.random.randint(low=0, high=10)
        self.img_gt = self.images[self.num_gt]
        if self.tactile_sim:
            # make sure the pixel value at initial location is not white. not starting on / within boundaries
            indices = np.transpose(np.where(self.img_gt != mu.white))
            initial_loc = tuple(indices[random.choice(range(indices.shape[0]))])
        else:
            initial_loc = tuple(np.random.randint(low=(0, 0), high=(28, 28)))
        
        self.current_step = 0

        pixel_value = self.img_gt[initial_loc]
        self.img_belief[0][initial_loc] = mu.current_black if pixel_value == mu.black else mu.current_white
        self.current_loc = initial_loc

        if self.ob_type == 'local':
            ob = np.array([pixel_value, initial_loc[0], initial_loc[1]])
        elif self.ob_type == 'global':
            ob = self.img_belief
        else:
            raise TypeError
        return ob


    def step(self, action):
        if self.action_type == 'only_explore':
            num_explored = np.count_nonzero(self.img_belief != mu.unexplored)
            new_loc = self.compute_next_loc(action)
            pixel_value = self.img_gt[new_loc]
            if self.tactile_sim and pixel_value == mu.white:
                self.img_belief[0][new_loc] = mu.white
                new_loc = self.current_loc
            else:
                # change the pixel at current location assuming the agent has left
                self.img_belief[0][self.current_loc] = mu.black if self.img_belief[0][self.current_loc] \
                                                                    == mu.current_black else mu.white
                # reveal pixel at new location, assuming the agent is on the new location
                self.img_belief[0][new_loc] = mu.current_black if pixel_value == mu.black else mu.current_white
            self.discover = True if np.count_nonzero(self.img_belief != mu.unexplored) > num_explored else False
            self.current_step += 1
            self.current_loc = new_loc

            # compute observation
            if self.ob_type == 'local':
                ob = np.array([pixel_value, new_loc[0], new_loc[1]])
            elif self.ob_type == 'global':
                ob = self.img_belief
            else:
                raise TypeError

            # discriminator does not care about agent location
            discriminator_input = copy.deepcopy(self.img_belief)
            discriminator_input[0][self.current_loc] = mu.black if discriminator_input[0][self.current_loc] == \
                                                                        mu.current_black else mu.white
            self.prediction, self.max_prob, self.probs = self.discriminator.predict(discriminator_input)
            # print('prediction: {} \n max_prob: {} \n probs: {}'.format(prediction, max_prob, probs))
            self.info = {}
            self.done = self.check_done()
            self.success = self.check_success()
            self.info.update({'success': self.success,
                                'num_explored_pixels': np.count_nonzero(self.img_belief != 127),
                                'prediction': self.prediction,
                                'max_prob': self.max_prob,
                                'num_gt': self.num_gt,
                                'discriminator_input': discriminator_input,
                                'discover': self.discover})

            reward = self.compute_reward()
            return ob, reward, self.done, self.info
        
        elif self.action_type == 'explore_and_predict':
            if action <= 3:
                self.img_belief[0][self.current_loc] = mu.black \
                    if self.img_belief[0][self.current_loc] == mu.current_black else mu.white
                new_loc = self.compute_next_loc(action)
                pixel_value = self.img_gt[new_loc]
                discover = True if self.img_belief[0][new_loc] == mu.unexplored else False
                self.img_belief[0][new_loc] = mu.current_black if pixel_value == mu.black else mu.current_white
                self.current_step += 1
                self.current_loc = new_loc

                # compute observation
                if self.ob_type == 'local':
                    ob = np.array([pixel_value, new_loc[0], new_loc[1]])
                elif self.ob_type == 'global':
                    ob = self.img_belief
                else:
                    raise TypeError

                done = True if self.current_step == self.max_ep_len else False
                reward = 0
                info = {}
                info.update({'success': False})
                return ob, reward, done, info
            
            else:
                ob = self.img_belief
                prediction = action - 4
                done = True
                info = {}
                if prediction == self.num_gt:
                    reward = 1
                    info.update({'success': True})
                else:
                    reward = 0
                    info.update({'success': False})
                return ob, reward, done, info

    def check_done(self):
        return self.current_step == self.max_ep_len or self.max_prob >= self.done_threshold

    def check_success(self):
        return self.max_prob >= self.done_threshold and self.prediction == self.num_gt

    def compute_reward(self):
        if self.reward_type == 'only_done_reward':
            # only penalize time, fully trust discriminator, learn to satisfy discriminator
            if self.done:
                return 1
            else:
                return 0
        elif self.reward_type == 'only_success_reward':
            if self.success:
                return 1.0
            else:
                return 0.0
        elif self.reward_type == 'new_pixel':
            if self.discover:
                return 1.0
            else:
                return 0.0
        else:
            raise TypeError

    def render(self, mode='human'):
        if not self.rendered:
            cv2.namedWindow('image'+str(self.window_id), cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image'+str(self.window_id), 300, 300)
            self.rendered = True
        if mode == 'rgb_array':
            return self.img_belief  # return RGB frame suitable for video
        elif mode == 'human':
            # pop up a window for visualization
            cv2.imshow('image'+str(self.window_id), self.img_belief[0])
            cv2.waitKey(1)
            return self.img_belief
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


