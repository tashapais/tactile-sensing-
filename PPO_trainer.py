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


HEIGHT = WIDTH = 32

class PPO_trainer():
    def __init__(self, num_parralel_envs, num_total_timesteps, num_steps, gae_lambda, num_minibatches=4,learning_rate=1e-3, action_dim=4, anneal_lr=False, multiprocess=False, gamma=0.95, epochs=4, clip_coef=0.1, clip_vloss=True, entropy_coef=0.05, value_coef=0.5, max_grad_norm=0.5, terminal_confidence=0.95):
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
        
    def rollout_no_parralelization(self):
        pass

    def make_env(seed):
        def thunk():
            env = GridWorldEnv()
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    def general_advantage_estimator(self):
        pass

    def training_actor(self):
        pass

    def training_critic(self):
        pass

    def optimize_policy_value(self):
        pass
    

    def train(self, discriminator):
        envs = VecPyTorch(
            SubprocVecEnv([self.make_env(self.seed + i)
                           for i in range(self.num_parralel_envs)], "fork"), self.device)
    
        next_obs = torch.Tensor(envs.reset()).to(self.device)
        next_done = torch.zeros(self.num_parralel_envs).to(self.device)

        for update_num in range(self.num_updates):
            if self.anneal_lr:
                frac = 1.0 - (update_num - 1.0) / self.num_updates
                self.lr = self.lr*frac
                self.optimizer.param_groups[0]["lr"] = self.lr

            for step in range(self.num_steps):
                self.global_step += self.num_parralel_envs
                self.obs[step] = next_obs   
                self.dones[step] = next_done

                with torch.no_grad():
                    move, logprob, _ = self.agent.get_action_and_move(next_obs, discriminator)
                    value = self.agent.get_value(next_obs)

                self.values[step] = value.flatten()
                self.logprobs[step] = logprob
                self.moves[step] = move

                prediction, max_prob, probs = discriminator.predict(self.obs[step].cpu().numpy())

                action = [{'move': move[i].item(),
                           'prediction': prediction[i],
                           'max_prob': max_prob[i],
                           'probs': probs[i],
                           'done': 1 if max_prob[i] >= self.terminal_confidence else 0
                           } for i in range(self.num_parralel_envs)]


                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)     
            

            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
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

            
            batch_obs = self.obs.reshape((-1,) + self.single_observation_space_shape)
            batch_logprobs = self.logprobs.reshape(-1)
            batch_moves = self.moves.reshape((-1,) + self.single_action_space_shape)
            batch_advantages = advantages.reshape(-1)
            batch_returns = returns.reshape(-1)
            batch_values = self.values.reshape(-1)


            batch_indices = np.arange(self.batch_size)
            clipfracs = []

            for epoch in range(self.epochs):
                batch_order = np.random.permutation(batch_indices)

                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    minibatch_indices = batch_order[start:end]
                    minibatch_advantages = batch_advantages[minibatch_indices]

                    _, newlogprob, entropy = agent.get_move(batch_obs[minibatch_indices], batch_moves.long()[minibatch_indices])
                    entropy_loss = entropy.mean()

                    logratio = newlogprob - batch_logprobs[minibatch_indices]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    # Policy loss
                    pg_loss1 = -minibatch_advantages * ratio
                    pg_loss2 = -minibatch_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()


                    # Value loss
                    new_values = agent.get_value(batch_obs[minibatch_indices]).view(-1)
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
                    nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    
if __name__ == "__main__":
    p = PPO_trainer()
    agent = Agent(action_dim=4, device=p.device)
    p.rollout_no_parralelization(agent=agent)