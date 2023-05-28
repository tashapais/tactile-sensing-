from discriminator_dataset import ImageDataset
import torch
from ppo_discrete import Agent
import tqdm
from grid_world_env_torch import GridWorldEnv
from data import DataLoader
import matplotlib.pyplot as plt

HEIGHT, WIDTH = 32, 32
NUM_EPISODES = 1000
MAX_EP_LEN = 5000
BUFFER_SIZE = 1000

class CoTrainingAlgorithm():
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.dataloader = DataLoader(batch_size=1)
        
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
            while not done:
                action, log_prob, entropy = agent.get_move(img)
                done, img = grid_world_env.step(action)
                img = img.to(self.device)
                cifar_dataset.add_data(img, label)
                pbar.update(1)
                if len(cifar_dataset) == cifar_dataset.buffer_size:
                        break
        pbar.close()

        return cifar_dataset


if __name__ == "__main__":
    algo = CoTrainingAlgorithm()
    dataset = algo.generate_training_data()