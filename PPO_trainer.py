from discriminator_dataset import ImageDataset
import torch
from ppo_discrete import Agent
import tqdm
from grid_world_env_torch import GridWorldEnv
from data import DataLoader
import matplotlib.pyplot as plt
import misc_utils as mu 
from logger import logger

HEIGHT = WIDTH = 32

class PPO_trainer():
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def rollout_no_parralelization(self, agent):
        EPISODES = 1

        for _ in range(EPISODES):
            cifar_dataset = ImageDataset(buffer_size=100,height=HEIGHT,width=WIDTH)
            print(len(cifar_dataset))
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