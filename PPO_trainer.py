from discriminator_dataset import ImageDataset
import torch
from ppo_discrete import Agent
import tqdm
from grid_world_env_torch import GridWorldEnv
from data import DataLoader
import matplotlib.pyplot as plt
import misc_utils as mu 
from logger import logger
from torch import optim 

HEIGHT = WIDTH = 32

class PPO_trainer():
    def __init__(self, num_parralel_envs, num_total_timesteps, num_steps, num_minibatches=4,learning_rate=1e-3,action_dim):
        #training params
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.action_dim = action_dim
        self.agent = Agent(action_dim=self.action_dim, device=self.device)
        self.num_parralel_envs = num_parralel_envs
        self.num_total_timesteps = num_total_timesteps
        self.num_steps = num_steps
        self.num_minibatches = num_minibatches
        self.batch_size = int(num_steps*num_parralel_envs)
        self.minibatch_size = int(self.batch_size//self.num_minibatches)
        self.num_updates = num_total_timesteps//self.batch_size
        self.lr = learning_rate
        self.optimizer = optim.Adam(agent.parameters(), lr=self.lr, eps=1e-5) 
        self.global_step = 0
        
        #env parameters
        self.height = HEIGHT
        self.width = WIDTH
        self.single_observation_space_shape = (HEIGHT, WIDTH)
        self.single_action_space_shape = (self.action_dim,)

        #storage params
        self.obs = torch.zeros((self.num_steps, self.num_parralel_envs) + self.single_observation_space_shape).to(self.device)
        self.actions = torch.zeros((self.num_steps, self.num_parralel_envs) + self.single_action_space_shape).to(self.device)
        self.logprobs = torch.zeros((self.num_steps, self.num_parralel_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_parralel_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_parralel_envs)).to(self.device)
        self.values = torch.zeros((self.num_steps, self.num_parralel_envs)).to(self.device)
        
    def rollout_no_parralelization(self):
        pass

    def general_advantage_estimator(self):
        pass

    def training_actor(self):
        pass

    def training_critic(self):
        pass
    


if __name__ == "__main__":
    p = PPO_trainer()
    agent = Agent(action_dim=4, device=p.device)
    p.rollout_no_parralelization(agent=agent)