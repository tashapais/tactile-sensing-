import torch 
import torch.nn as nn 
import torch.nn.functional as F
import tqdm 
import torch.optim as optim
import os



"""The discriminator works on a partially revealed image to understand where to go next. 
"""



class Discriminator_NN(nn.Module):
    def __init__(self, height, width, save_dir):
        super(Discriminator_NN, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.pool = nn.MaxPool2d((2, 2))
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax()

        self.model = nn.Sequential(
                    self.conv1, 
                    self.ReLU,
                    self.conv2,
                    self.ReLU,
                    self.fc1, 
                    self.ReLU, 
                    self.fc2, 
                    self.ReLU,
                    self.fc3)
        self.lr = 0.001
        self.save_dir = save_dir

    def forward_logprob(self, x):
        # Max pooling over a (2, 2) window 
        # If the size is a square, you can specify with a single number 
        x = self.model(x)
        output = F.log_softmax(x, dim=1)
        return output

    def forward_prob(self, x):
        x = self.forward(x)
        probabilities = F.softmax(x, dim=1)
        return probabilities

    def load_model(self, model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.model.to(self.device)
            print(f'model loaded from {model_path}')

    def train_epoch(self, epoch, data_loader, logger=None):
        log = logger.log if logger is not None else print
        pbar = tqdm.tqdm(total=len(data_loader.dataset))
        self.model.train()
        correct = 0
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)
            data = data.float()
            self.optimizer.zero_grad()
            output = self.forward_logprob(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            # item() is important here for saving memory
            # https://discuss.pytorch.org/t/cpu-ram-usage-increasing-for-every-epoch/24475/6
            epoch_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            pbar.update(data.shape[0])

        pbar.close()
        epoch_loss = epoch_loss / len(data_loader.dataset)
        acc = correct / len(data_loader.dataset)
        log('Train Epoch: {} | Loss: {:.6f} | Acc: {:.6f}'.format(epoch, epoch_loss, acc))

        return epoch_loss, acc

    def test_epoch(self, epoch, data_loader, logger=None):
        log = logger.log if logger is not None else print
        pbar = tqdm.tqdm(total=len(data_loader.dataset))
        self.model.eval()
        correct = 0
        epoch_loss = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.float()
                output = self.forward_logprob(data)
                epoch_loss += F.nll_loss(output, target, reduction='sum')
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                pbar.update(data.shape[0])

        pbar.close()
        epoch_loss = epoch_loss.item() / len(data_loader.dataset)
        acc = correct / len(data_loader.dataset)
        log('Test Epoch: {} | Loss: {:.6f} | Acc: {:.6f}\n'.format(epoch, epoch_loss, acc))
        self.loss = epoch_loss
        return epoch_loss, acc
    
    def learn(self, epochs, train_loader, test_loader, use_best_model=True, logger=None):
        log = logger.log if logger is not None else print 
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        stats = []
        # the stat for the best model
        model_path = None
        test_acc = None
        test_loss = None
        train_loss = None
        train_acc = None
        max_test_acc = 0

        for i in range(epochs):
            train_loss, train_acc = self.train_epoch(i, train_loader, logger)
            test_loss, test_acc = self.test_epoch(i, test_loader, logger)
            # self.scheduler.step()

            stats.append({
                'epoch': i,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc
            })

            # save dir is changing with different batch of data collected for discriminator
            if self.save_dir is not None and test_acc >= max_test_acc:
                # for each epoch
                model_folder_name = f'epoch_{i:06d}_loss_{test_loss:.6f}_acc_{test_acc:.8f}'
                if not os.path.exists(os.path.join(self.save_dir, model_folder_name)):
                    os.makedirs(os.path.join(self.save_dir, model_folder_name))
                model_path = os.path.join(self.save_dir, model_folder_name, 'model.pth')
                torch.save(self.model.state_dict(), model_path)
                log(f'model saved to {model_path}\n')
                stats[i]['model_path'] = model_path
                max_test_acc = test_acc

        if use_best_model and self.save_dir is not None:
            # return the best model according to the testing accuracy, this will pick the later one with equal acc
            best_stat = sorted(stats, key=lambda x: x['test_acc'])[-1]
            model_path, test_acc, test_loss, train_loss, train_acc = \
                best_stat['model_path'], best_stat['test_acc'], best_stat['test_loss'], best_stat['train_loss'], best_stat['train_acc']
            self.load_model(model_path=model_path)
            log(f're-loaded model path {model_path}')

        return model_path, train_loss, train_acc, test_loss, test_acc, stats