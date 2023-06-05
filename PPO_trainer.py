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
import torch.nn as nn
import gym
from ppo_discrete import Agent, VecPyTorch, linear_schedule
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
import time 
import numpy as np
from ppo_from_scratch_squared import CoTrainingAlgorithm


HEIGHT = WIDTH = 32

class PPO_trainer(CoTrainingAlgorithm):
    def __init__(self):        
        #training params
        self.device = CoTrainingAlgorithm.device
        self.action_dim = CoTrainingAlgorithm.action_dim
        self.agent = CoTrainingAlgorithm.agent
        self.num_parralel_envs = CoTrainingAlgorithm.num_parralel_envs
        self.num_total_timesteps = CoTrainingAlgorithm.num_total_timesteps
        self.num_steps = CoTrainingAlgorithm.num_steps
        self.num_minibatches = CoTrainingAlgorithm.num_minibatches
        self.batch_size = CoTrainingAlgorithm.batch_size
        self.minibatch_size = CoTrainingAlgorithm.minibatch_size
        self.num_updates = CoTrainingAlgorithm.num_updates
        self.lr = CoTrainingAlgorithm.lr
        self.optimizer = CoTrainingAlgorithm.optimizer
        self.global_step = CoTrainingAlgorithm.global_step
        self.anneal_lr = CoTrainingAlgorithm.anneal_lr
        self.multiprocess = CoTrainingAlgorithm.multiprocess
        self.gamma = CoTrainingAlgorithm.gamma
        self.gae_lambda = CoTrainingAlgorithm.gae_lambda
        self.epochs = CoTrainingAlgorithm.epochs
        self.clip_coef = CoTrainingAlgorithm.clip_coef
        self.clip_vloss = CoTrainingAlgorithm.clip_vloss
        self.entropy_coef = CoTrainingAlgorithm.entropy_coef
        self.value_coef = CoTrainingAlgorithm.value_coef
        self.max_grad_norm = CoTrainingAlgorithm.max_grad_norm
        self.terminal_confidence = CoTrainingAlgorithm.terminal_confidence
        self.discriminator = CoTrainingAlgorithm.discriminator

        #env parameters
        self.height = CoTrainingAlgorithm.height
        self.width = CoTrainingAlgorithm.width
        self.single_observation_space_shape = CoTrainingAlgorithm.single_observation_space_shape
        self.single_action_space_shape = CoTrainingAlgorithm.single_action_space_shape
        self.seed = CoTrainingAlgorithm.seed

        
        #storage params
        self.obs = CoTrainingAlgorithm.obs
        self.moves = CoTrainingAlgorithm.moves
        self.logprobs = CoTrainingAlgorithm.logprobs
        self.rewards = CoTrainingAlgorithm.rewards
        self.dones = CoTrainingAlgorithm.dones
        self.values = CoTrainingAlgorithm.values
        

    def make_env(self, seed):
        def thunk():
            env = GridWorldEnv(max_ep_len=, image=, label=)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    def rollout(self):
        for step in range(self.num_steps):
            self.global_step += self.num_parralel_envs
            self.obs[step] = next_obs   
            self.dones[step] = next_done

            with torch.no_grad():
                move, logprob, _ = self.agent.get_action_and_move(next_obs, self.discriminator)
                value = self.agent.get_value(next_obs)

            self.values[step] = value.flatten()
            self.logprobs[step] = logprob
            self.moves[step] = move

            prediction, max_prob, probs = self.discriminator.predict(self.obs[step].cpu().numpy())

            action = [{'move': move[i].item(),
                        'prediction': prediction[i],
                        'max_prob': max_prob[i],
                        'probs': probs[i],
                        'done': 1 if max_prob[i] >= self.terminal_confidence else 0
                        } for i in range(self.num_parralel_envs)]


            next_obs, reward, done, info = self.envs.step(action.cpu().numpy())
            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)
            
        return next_obs, next_done
    

    def advantage_return_computation(self, next_obs, next_done):
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values

        return advantages, returns



    def optimization(self, batch_obs, batch_logprobs, batch_moves, batch_advantages, batch_returns, batch_values):
        batch_indices = np.arange(self.batch_size)
        clipfracs = []

        for _ in range(self.epochs):
            batch_order = np.random.permutation(batch_indices)

            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                minibatch_indices = batch_order[start:end]
                minibatch_advantages = batch_advantages[minibatch_indices]

                _, newlogprob, entropy = self.agent.get_move(batch_obs[minibatch_indices], batch_moves.long()[minibatch_indices])
                entropy_loss = entropy.mean()

                logratio = newlogprob - batch_logprobs[minibatch_indices]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                # Policy loss
                pg_loss1 = -minibatch_advantages * ratio
                pg_loss2 = -minibatch_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()


                # Value loss
                new_values = self.agent.get_value(batch_obs[minibatch_indices]).view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = ((new_values - batch_returns[minibatch_indices]) ** 2)
                    v_clipped = batch_values[minibatch_indices] + torch.clamp(new_values - batch_values[minibatch_indices],
                                                                    -self.clip_coef, self.clip_coef)
                    v_loss_clipped = (v_clipped - batch_returns[minibatch_indices]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_values - batch_returns[minibatch_indices]) ** 2).mean()


                loss = pg_loss - self.entropy_coef * entropy_loss + v_loss * self.value_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()
    

    def make_envs(self):

        pass




    def train(self):


        self.envs = VecPyTorch(
            SubprocVecEnv([self.make_env(self.seed + i)
                           for i in range(self.num_parralel_envs)], "fork"), self.device)
    
        next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        next_done = torch.zeros(self.num_parralel_envs).to(self.device)

        for update_num in range(1, self.num_updates+1):
            self.discriminator.train_discriminator()

            if self.anneal_lr:
                frac = 1.0 - (update_num - 1.0) / self.num_updates
                self.lr = self.lr*frac
                self.optimizer.param_groups[0]["lr"] = self.lr

            next_obs, next_done = self.rollout()
            advantages, returns = self.advantage_return_computation(next_obs=next_obs, next_done=next_done)
            

            
            batch_obs = self.obs.reshape((-1,) + self.single_observation_space_shape)
            batch_logprobs = self.logprobs.reshape(-1)
            batch_moves = self.moves.reshape((-1,) + self.single_action_space_shape)
            batch_advantages = advantages.reshape(-1)
            batch_returns = returns.reshape(-1)
            batch_values = self.values.reshape(-1)


            self.optimization(batch_obs=batch_obs, 
                              batch_logprobs=batch_logprobs,
                              batch_moves=batch_moves, 
                              batch_advantages=batch_advantages, 
                              batch_returns=batch_returns, 
                              batch_values=batch_values)
                                
if __name__ == "__main__":
    p = PPO_trainer()