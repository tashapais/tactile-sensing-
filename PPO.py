from discriminator_dataset import ImageDataset
import torch
import tqdm
from grid_world_env import GridWorldEnv
from data import CIFARDataLoader
import misc_utils as mu
import torch.optim as optim
import time as time
from ppo_discrete import Agent, VecPyTorch
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import numpy as np
import torch.nn as nn
import gym
from pprint import pprint

HEIGHT, WIDTH = 32, 32
MAX_EP_LEN = 1000
BUFFER_SIZE = int(4e5)
COUNTER = 200


class CoTrainingAlgorithm:
    def __init__(self,
                 num_parallel_envs,
                 num_total_timesteps,
                 num_steps,
                 gae_lambda=0.95,
                 num_minibatches=4,
                 learning_rate=1e-3,
                 action_dim=4,
                 anneal_lr=False,
                 multiprocess=True,
                 gamma=0.95,
                 explorer_epochs=3,
                 discriminator_epochs=3,
                 clip_coef=0.1,
                 clip_vloss=True,
                 entropy_coef=0.05,
                 value_coef=0.5,
                 max_grad_norm=0.5,
                 terminal_confidence=0.95):

        # training params
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.dataloader = CIFARDataLoader(batch_size=1)
        self.discriminator_dataset = None
        self.discriminator = mu.construct_discriminator(discriminator_type="learned", height=HEIGHT, width=WIDTH,
                                                        lr=0.001)
        self.action_dim = action_dim
        self.num_parallel_envs = num_parallel_envs  # if multiprocess else 1
        self.agent = Agent(action_dim=self.action_dim, device=self.device, num_envs=self.num_parallel_envs)
        self.num_total_timesteps = num_total_timesteps
        self.num_steps = num_steps
        self.num_minibatches = num_minibatches
        self.batch_size = int(num_steps * num_parallel_envs)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_updates = int(num_total_timesteps // self.batch_size)
        self.lr = learning_rate
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.lr, eps=1e-5)
        self.global_step = 0
        self.anneal_lr = anneal_lr
        self.multiprocess = multiprocess
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.explorer_epochs = explorer_epochs
        self.discriminator_epochs = discriminator_epochs
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.terminal_confidence = terminal_confidence

        # env parameters
        self.height = HEIGHT
        self.width = WIDTH
        self.single_observation_space_shape = (3, HEIGHT, WIDTH)
        self.single_action_space_shape = (self.action_dim,)
        self.seed = int(time.time())
        self.env_images = iter(self.dataloader.return_trainloader())
        if self.multiprocess:
            # could this have anything to do with the device at all????
            # few things to try: 1. us the sub proc vec env with one if self.num_parralel_envs = 1, simple script to test out a simple env, and test out subprocvecenv class
            # go to github to stable baselines => got to stable baselines and search in their issues you might.
            # write a script to reproduce the error create a simple script that recreates the error.
            self.envs = VecPyTorch(
                SubprocVecEnv([self.make_env(self.seed + i) for i in range(self.num_parallel_envs)], 'fork'),
                self.device)
        else:
            self.envs = VecPyTorch(DummyVecEnv([self.make_env(self.seed + i) for i in range(self.num_parallel_envs)]),
                                   self.device)
        # storage params
        self.obs = torch.zeros((self.num_steps, self.num_parallel_envs) + self.envs.observation_space.shape).to(
            self.device)
        self.moves = torch.zeros((self.num_steps, self.num_parallel_envs) + self.envs.action_space['move'].shape).to(
            self.device)
        self.logprobs = torch.zeros((self.num_steps, self.num_parallel_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_parallel_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_parallel_envs)).to(self.device)
        self.values = torch.zeros((self.num_steps, self.num_parallel_envs)).to(self.device)

    def generate_training_data(self):
        cifar_dataset = ImageDataset(buffer_size=BUFFER_SIZE, height=HEIGHT, width=WIDTH)
        agent = Agent(action_dim=4, device=self.device, num_envs=1)
        pbar = tqdm.tqdm(total=cifar_dataset.buffer_size)

        mx = COUNTER
        count = 0
        for original_image, label in self.env_images:
            if count == mx:
                break

            count += 1

            grid_world_env = GridWorldEnv(max_ep_len=MAX_EP_LEN,
                                          label=label[0],
                                          image=original_image[0])
            img = grid_world_env.reset()
            img = img.to(self.device)
            done = False
            while not done and not len(cifar_dataset) == cifar_dataset.buffer_size:
                action, log_prob, entropy = agent.get_move(torch.unsqueeze(img, 0))
                done, img = grid_world_env.step(action)
                img = img.to(self.device)
                cifar_dataset.add_data(torch.unsqueeze(img, dim=0), label)
                pbar.update(1)
        pbar.close()
        self.discriminator_dataset = cifar_dataset

    def train_discriminator(self):
        if self.discriminator_dataset is not None:
            train_loader, test_loader = mu.construct_loaders(self.discriminator_dataset, split=0.2)
            discriminator_path, discriminator_train_loss, discriminator_train_acc, discriminator_test_loss, discriminator_test_acc, \
                stats = self.discriminator.learn(epochs=self.discriminator_epochs, train_loader=train_loader,
                                                 test_loader=test_loader)
            pprint(stats)
        else:
            raise Exception("Discriminator dataset not configured yet")

    def make_env(self, seed):
        def thunk():
            img, label = next(self.env_images)
            env = GridWorldEnv(image=img[0], label=label[0], max_ep_len=self.num_steps)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        return thunk

    def rollout(self, next_obs, next_done):
        for step in range(self.num_steps):
            self.global_step += self.num_parallel_envs
            self.obs[step] = next_obs
            self.dones[step] = next_done

            with torch.no_grad():
                move, logprob, _ = self.agent.get_move(next_obs)
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
                       } for i in range(self.num_parallel_envs)]

            next_obs, reward, dones, infos = self.envs.step(action)
            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(dones).to(self.device)

            self.add_new_data(infos, dones)

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

        for _ in range(self.explorer_epochs):
            batch_order = np.random.permutation(batch_indices)

            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                minibatch_indices = batch_order[start:end]
                minibatch_advantages = batch_advantages[minibatch_indices]

                _, newlogprob, entropy = self.agent.get_move(batch_obs[minibatch_indices],
                                                             batch_moves.long()[minibatch_indices])
                entropy_loss = entropy.mean()

                logratio = newlogprob - batch_logprobs[minibatch_indices]
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                # Policy loss
                pg_loss1 = -minibatch_advantages * ratio
                pg_loss2 = -minibatch_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_values = self.agent.get_value(batch_obs[minibatch_indices]).view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = ((new_values - batch_returns[minibatch_indices]) ** 2)
                    v_clipped = batch_values[minibatch_indices] + torch.clamp(
                        new_values - batch_values[minibatch_indices],
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

    def add_new_data(self, infos, dones):
        for (i, info, done) in zip(range(len(infos)), infos, dones):
            if info['discover']:
                img = info['img']
                self.discriminator_dataset.add_data(torch.unsqueeze(img, 0), [info['label']])

    def co_training_loop(self):
        next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        next_done = torch.zeros(self.num_parallel_envs).to(self.device)

        count = 0
        mx = COUNTER
        for update_num in range(1, self.num_updates + 1):
            if count > mx:
                break

            count += 1
            self.train_discriminator()

            if self.anneal_lr:
                frac = 1.0 - (update_num - 1.0) / self.num_updates
                self.lr = self.lr * frac
                self.optimizer.param_groups[0]["lr"] = self.lr

            next_obs, next_done = self.rollout(next_obs, next_done)
            advantages, returns = self.advantage_return_computation(next_obs=next_obs, next_done=next_done)

            batch_obs = self.obs.reshape((-1,) + self.envs.observation_space.shape)
            batch_logprobs = self.logprobs.reshape(-1)
            batch_moves = self.moves.reshape((-1,) + self.envs.action_space['move'].shape)
            batch_advantages = advantages.reshape(-1)
            batch_returns = returns.reshape(-1)
            batch_values = self.values.reshape(-1)

            self.optimization(batch_obs=batch_obs,
                              batch_logprobs=batch_logprobs,
                              batch_moves=batch_moves,
                              batch_advantages=batch_advantages,
                              batch_returns=batch_returns,
                              batch_values=batch_values)

    def save_models(self):
        DIR = "./SAVED_MODELS"
        torch.save(self.agent.state_dict(), DIR + "/AGENT")
        self.discriminator.save_model(DIR, "DISCRIMINATOR")


if __name__ == "__main__":
    co_trainer = CoTrainingAlgorithm(num_parallel_envs=1,
                                     num_total_timesteps=int(2e4),
                                     num_steps=MAX_EP_LEN,
                                     multiprocess=False)
    printXs = [print("X", end="\n") if i == 99 else print("X", end="") for i in range(100)]
    print("XXXXXX GENERATING TRAINING DATA XXXXXXXXX")
    printXs = [print("X", end="\n") if i == 99 else print("X", end="") for i in range(100)]
    print("\n")
    co_trainer.generate_training_data()

    printXs = [print("X", end="\n") if i == 99 else print("X", end="") for i in range(100)]
    print("XXXXXX TRAINING DISCRIMINATOR XXXXXXXXX")
    printXs = [print("X", end="\n") if i == 99 else print("X", end="") for i in range(100)]
    print("\n")

    co_trainer.train_discriminator()

    printXs = [print("X", end="\n") if i == 99 else print("X", end="") for i in range(100)]
    print("XXXXXX INITIATING COTRAINING LOOP XXXXXXXXX")
    printXs = [print("X", end="\n") if i == 99 else print("X", end="") for i in range(100)]
    print("\n")

    co_trainer.co_training_loop()

    printXs = [print("X", end="\n") if i == 99 else print("X", end="") for i in range(100)]
    print("XXXXXX SAVING MODELS XXXXXXXXX")
    printXs = [print("X", end="\n") if i == 99 else print("X", end="") for i in range(100)]
    print("\n")

    co_trainer.save_models()
