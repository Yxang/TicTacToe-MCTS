import multiprocessing
import logging
import queue
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import Env
from Agents import RandomAgent, MCTSAgent, NNAgent, MCTSNNAgent
import NNReferee


class NNDataset(Dataset):
    """
    the pytorch dataset that stores the data from MCTS process
    """
    def __init__(self, dataset):
        """
        :param dataset: a list
        """
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

    @staticmethod
    def collate_fn(samples):
        transposed_samples = list(zip(*samples))
        nn_features = torch.cat([item for item in transposed_samples[0]], dim=0)
        policy = torch.stack([item for item in transposed_samples[1]])
        value = torch.stack([item for item in transposed_samples[2]]).unsqueeze(-1)
        return nn_features, policy, value

    def to_device(self, item, device):
        nn_feature, policy, value = item
        nn_feature = nn_feature.to(device)
        policy, value = policy.to(device), value.to(device)
        return nn_feature, policy, value


class NNLoss(nn.Module):
    def forward(self, policy, policy_hat, value, value_hat):
        value_error = (value - value_hat) ** 2
        policy_error = torch.sum((-policy *
                                  (1e-8 + policy_hat.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error


class NNTrainer:
    def __init__(self, nn, training_data, cfg=None):
        self.nn = nn
        self.ds = NNDataset(training_data)
        self.cfg = cfg
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.criterion = NNLoss()
        self.optimizer = optim.Adam(self.nn.parameters(), weight_decay=3e-4)
        self.dl = DataLoader(self.ds, batch_size=2, shuffle=True, collate_fn=self.ds.collate_fn)
        self.loss_list = []

    def train(self, n_epoch=50):
        """
        train the network
        :param n_epoch: num of epoch
        """
        self.nn.train()
        for epoch in range(n_epoch):
            running_loss = 0.
            for i, item in tqdm(enumerate(self.dl)):
                self.optimizer.zero_grad()
                nn_feature, policy, value = self.ds.to_device(item, self.device)
                policy_hat, value_hat = self.nn(nn_feature)
                loss = self.criterion(policy, policy_hat, value, value_hat)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            self.loss_list.append(running_loss)
            print('epoch:%d loss: %.3f' %
                  (epoch + 1, running_loss / 2))

    def save(self, path):
        """
        save the network to the path
        :param path: path to the model
        """
        torch.save(self.nn.state_dict(), path)


def load_model(model_class, path):
    """
    load the model from the path
    :param model_class: the class of the model
    :param path: path to the model
    :return net: the network
    """
    net = model_class()
    net.load_state_dict(torch.load(path))
    return net


if __name__ == '__main__':
    net = NNAgent.NN()
    agent1 = {'agent': MCTSNNAgent.MCTSNNAgent, 'params': (1, net)}
    agent2 = {'agent': MCTSNNAgent.MCTSNNAgent, 'params': (-1, net)}
    referee = NNReferee.NNReferee()
    referee.setup(agent1, agent2, mt=False)
    result, training_data = referee.host()
    trainer = NNTrainer(net, training_data)
    trainer.train()
    trainer.save('models/test.model')
