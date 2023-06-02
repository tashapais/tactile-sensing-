from discriminator_dataset import ImageDataset
import torch
from ppo_discrete import Agent
import tqdm
from grid_world_env_torch import GridWorldEnv
from data import DataLoader
import matplotlib.pyplot as plt
import misc_utils as mu 
from logger import logger

HEIGHT, WIDTH = 32, 32
MAX_EP_LEN = 10
BUFFER_SIZE = 100

class CoTrainingAlgorithm():
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.dataloader = DataLoader(batch_size=1)
        self.discriminator_dataset = None
        self.discriminator = mu.construct_discriminator(discriminator_type="learned",
                                                   height=HEIGHT,
                                                   width=WIDTH,
                                                   lr=0.001)
        
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
        


    
    
    def train_rl_ppo_rollout(self):
        batch_size = MAX_EP_LEN
        length_dataset = 600000
        total_timesteps = length_dataset*MAX_EP_LEN
        #num_updates = total_timesteps//batch_size
        #Overly done logic; do in place

        cifar_dataset = ImageDataset(buffer_size=60000, height=HEIGHT, width=WIDTH)
        agent = Agent(action_dim=4, device=self.device)
        pbar = tqdm.tqdm(total=cifar_dataset.buffer_size)

        train_cifar_iterator = iter(self.dataloader.return_trainloader())

        for update in range(1, 2):
            self.train_discriminator()

            cifar_dataset = ImageDataset(buffer_size=60000, height=HEIGHT, width=WIDTH)
            agent = Agent(action_dim=4, device=self.device)
            pbar =  tqdm.tqdm(total=cifar_dataset.buffer_size)

            train_cifar_iterator = iter(self.dataloader.return_trainloader())

            original_image, label = next(train_cifar_iterator)
            grid_world_env = GridWorldEnv(max_ep_len=MAX_EP_LEN,
                                    label=label[0],
                                    image=original_image[0])
            
            states = torch.zeros((10,32,32))
            moves = torch.zeros((10))
            rewards = torch.zeros((10))
            logprobs = torch.zeros((10))
            values = torch.zeros((10))
            dones = torch.zeros((10))

            next_obs = grid_world_env.reset()
            next_done = torch.zeros(1).to(self.device)
            states.append(next_obs)

            for step in range(0, MAX_EP_LEN):
                states[step] = next_obs
                dones[step] = next_done


                with torch.no_grad():
                    values[step] = agent.get_value(states[step]).flatten()
                    move, logproba, _ = agent.get_move(states[step])
                
                moves[step] = move 
                logprobs[step] = logproba

                prediction, max_prob, probs = self.discriminator.predict(states[step].cpu().numpy())

                action = {'move':move, 
                          'prediction':prediction, 
                          'max_prob': max_prob,
                          'probs': probs,
                          'done': 1 if max_prob>0.95 else 0
                          }
            
            next_obs, next_done = grid_world_env.step(action)
            #now computing the advantage estimation 
            #computing the advantage function for each time step 

            with torch.no_grad():
                last_value = agent.get_value(next_obs.to(self.device))
                returns =  torch.zeros_like(rewards).to(self.device)
                for t in range(MAX_EP_LEN)[::-1]:
                    if t == MAX_EP_LEN - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = last_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values
            
            b_states = states.reshape((-1,) + (32,32))  # [1024, 1, 50, 50]
            b_logprobs = logprobs.reshape(-1)
            b_moves = moves.reshape((-1,) + (4, 1))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)








if __name__ == "__main__":
    algo = CoTrainingAlgorithm()
    dataset = algo.generate_training_data()
    algo.train_discriminator()