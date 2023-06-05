from discriminator_dataset import ImageDataset
import torch
from ppo_discrete import Agent
import tqdm
from grid_world_env_torch import GridWorldEnv
from data import DataLoader
import matplotlib.pyplot as plt
import misc_utils as mu 
from logger import logger
import torch.optim as optim
import time as time 
from ppo_discrete import Agent, VecPyTorch, linear_schedule
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
import numpy as np
import torch.nn as nn  
import gym
from PPO_trainer_2 import PPO_trainer



HEIGHT, WIDTH = 32, 32
MAX_EP_LEN = 10
BUFFER_SIZE = 100

class CoTrainingAlgorithm():
    def __init__(self,
                 num_parralel_envs, 
                 num_total_timesteps, 
                 num_steps, 
                 gae_lambda=0.95, 
                 num_minibatches=4,
                 learning_rate=1e-3, 
                 action_dim=4, 
                 anneal_lr=False, 
                 multiprocess=False, 
                 gamma=0.95, 
                 epochs=4, 
                 clip_coef=0.1, 
                 clip_vloss=True, 
                 entropy_coef=0.05, 
                 value_coef=0.5, 
                 max_grad_norm=0.5, 
                 terminal_confidence=0.95):
        #training params
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.dataloader = DataLoader(batch_size=1)
        self.discriminator_dataset = None
        self.discriminator = mu.construct_discriminator(discriminator_type="learned",
                                                   height=HEIGHT,
                                                   width=WIDTH,
                                                   lr=0.001)
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
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.lr, eps=1e-5) 
        self.global_step = 0
        self.anneal_lr = anneal_lr
        self.multiprocess = multiprocess
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.terminal_confidence = terminal_confidence
       
        
        #env parameters
        self.height = HEIGHT
        self.width = WIDTH
        self.single_observation_space_shape = (HEIGHT, WIDTH)
        self.single_action_space_shape = (self.action_dim,)
        self.seed = time.time()

        #storage params
        self.obs = torch.zeros((self.num_steps, self.num_parralel_envs) + self.single_observation_space_shape).to(self.device)
        self.moves = torch.zeros((self.num_steps, self.num_parralel_envs) + self.single_action_space_shape).to(self.device)
        self.logprobs = torch.zeros((self.num_steps, self.num_parralel_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_parralel_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_parralel_envs)).to(self.device)
        self.values = torch.zeros((self.num_steps, self.num_parralel_envs)).to(self.device)
        
    def generate_training_data(self):
        cifar_dataset = ImageDataset(buffer_size=BUFFER_SIZE, height=HEIGHT, width=WIDTH)
        agent = Agent(action_dim=4, device=self.device)
        pbar = tqdm.tqdm(total=cifar_dataset.buffer_size)

        train_cifar_iterator = iter(self.dataloader.return_trainloader())
        while len(cifar_dataset)<cifar_dataset.buffer_size: 
            original_image, label = next(train_cifar_iterator)
            grid_world_env = GridWorldEnv(max_ep_len=MAX_EP_LEN,
                                    label=label[0],
                                    image=original_image[0])
            img, ob = grid_world_env.reset()
            img = img.to(self.device)
            done = False
            while not done and not len(cifar_dataset) == cifar_dataset.buffer_size:
                action, log_prob, entropy = agent.get_move(img)
                done, img = grid_world_env.step(action)
                img = img.to(self.device)
                cifar_dataset.add_data(torch.unsqueeze(img,dim=0), label)
                pbar.update(1)
        pbar.close()
        self.discriminator_dataset = cifar_dataset
    
    def train_discriminator(self):
        if self.discriminator_dataset != None:
            train_loader, test_loader = mu.construct_loaders(self.discriminator_dataset, split=0.2)
            discriminator_path, discriminator_train_loss, discriminator_train_acc, discriminator_test_loss, discriminator_test_acc, stats = self.discriminator.learn(
                epochs=15,
                train_loader=train_loader,
                test_loader=test_loader)
            print(stats)
        else:
            raise Exception("Discriminator dataset not configured yet")
    
    
    def co_training_loop(self, ppo_trainer):
        envs = ppo_trainer.envs
    
        next_obs = torch.Tensor(envs.reset()).to(self.device)
        next_done = torch.zeros(self.num_parralel_envs).to(self.device)

        for update_num in range(1, self.num_updates+1):
            self.discriminator.train_discriminator()

            if self.anneal_lr:
                frac = 1.0 - (update_num - 1.0) / self.num_updates
                self.lr = self.lr*frac
                self.optimizer.param_groups[0]["lr"] = self.lr

            next_obs, next_done = ppo_trainer.rollout()
            advantages, returns = ppo_trainer.advantage_return_computation(next_obs=next_obs, next_done=next_done)
            

            
            batch_obs = self.obs.reshape((-1,) + self.single_observation_space_shape)
            batch_logprobs = self.logprobs.reshape(-1)
            batch_moves = self.moves.reshape((-1,) + self.single_action_space_shape)
            batch_advantages = advantages.reshape(-1)
            batch_returns = returns.reshape(-1)
            batch_values = self.values.reshape(-1)


            ppo_trainer.optimization(batch_obs=batch_obs, 
                                    batch_logprobs=batch_logprobs,
                                    batch_moves=batch_moves, 
                                    batch_advantages=batch_advantages, 
                                    batch_returns=batch_returns, 
                                    batch_values=batch_values)
            
        add_data()???

if __name__ == "__main__":
    co_trainer = CoTrainingAlgorithm(num_parralel_envs=4,num_total_timesteps=1e5, num_steps=MAX_EP_LEN)
    co_trainer.generate_training_data()
    co_trainer.train_discriminator()
    ppo_trainer = PPO_trainer(co_trainer)
