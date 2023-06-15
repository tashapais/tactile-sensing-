import torch.nn as nn
import misc_utils as mu
import numpy as np
from ppo_discrete import Agent


'''
Discriminator training, training_discriminator is done in two main steps. 
''' 
'''Explorer takes in observations and decides where to move next.'''


class Explorer_NN(nn.Module):

    def __init__(self, action_dim, device, model_path=None, frames=1, img_size=32):
        super(Explorer_NN, self).__init__()
        self.img_size = img_size
        self.agent = Agent(action_dim=action_dim,device=device,frames=frames, img_size=img_size)
        self.model_path = model_path

    def get_move(self, obs):
        move, probs, entropy = self.agent.get_move_stochastic(obs)
        next_loc = mu.compute_next_loc(mu.get_current_loc(obs[0]), move, height=self.img_size, width=self.img_size)
        index = 0
        while obs[0][next_loc] != mu.unexplored and obs[1][next_loc] != mu.unexplored and obs[2][next_loc] != mu.unexplored:
            # move = mu.get_next_direction_clockwise(move)
            move = sorted(zip(probs, range(4)), reverse=True, key=lambda x: x[0])[index][1]
            next_loc = mu.compute_next_loc(mu.get_current_loc(obs), move, height=self.img_size, width=self.img_size)
            index += 1
            # collision checking false, all neighbours are white
            if index == 4:
                rando = np.random.choice(4)
                return rando, probs.log_prob(rando), entropy
        return move, probs.log_prob(move), entropy

    def get_move_bkup(self, obs):
        if obs.ndim == 3:
            probs = self.agent.get_move_probabilities(obs[None, ...])[0]
            for prob, move in sorted(zip(probs, range(4)), reverse=True):
                next_loc = mu.compute_next_loc(mu.get_current_loc(obs), move, height=self.img_size, width=self.img_size)
                if obs[0][next_loc] != mu.white and obs[0][next_loc] != mu.black and obs[0][next_loc] != mu.current_black:
                    return move
            for prob, move in sorted(zip(probs, range(4)), reverse=True):
                next_loc = mu.compute_next_loc(mu.get_current_loc(obs), move, height=self.img_size, width=self.img_size)
                if obs[0][next_loc] != mu.white and obs[0][next_loc] != mu.current_black:
                    return move
        elif obs.ndim == 4:
            probs = self.agent.get_move_probabilities(obs)
