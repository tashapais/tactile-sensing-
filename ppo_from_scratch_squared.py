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
NUM_EPISODES = 1000
MAX_EP_LEN = 5000
BUFFER_SIZE = 1000

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
                test_loader=test_loader,
                logger=logger)
            print(stats)
        else:
            raise Exception("Discriminator dataset not configured yet")



if __name__ == "__main__":
    algo = CoTrainingAlgorithm()
    dataset = algo.generate_training_data()
    algo.train_discriminator()