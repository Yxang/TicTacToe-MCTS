import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

import Agents.MCTSAgent as MCTSAgent
import Env


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 5)
        self.fc2 = nn.Linear(9, 5)
        self.policy = nn.Linear(5, 9)
        self.value = nn.Linear(5, 1)

    def forward(self, x):
        """
        the input of this network is a (env, who) pair
        :param x:
        :return:
        """
        x, who = x
        x1 = (torch.where(x == who,
                          torch.tensor(1., device=get_nn_device(self)),
                          torch.tensor(0., device=get_nn_device(self)))
              .flatten(1)
              )
        x2 = (torch.where(x == -who,
                          torch.tensor(1., device=get_nn_device(self)),
                          torch.tensor(0., device=get_nn_device(self)))
              .flatten(1)
              )
        x1 = f.leaky_relu(self.fc1(x1))
        x2 = f.leaky_relu(self.fc1(x2))
        x = x1 + x2
        policy = f.softmax(self.policy(x), 1)
        value = torch.sigmoid(self.value(x)) * 2 - 1
        return policy, value


def convert_env_to_input(env, who):
    """
    convert a env (np.ndarray) to pytorch tensor
    :param env: the env to be converted
    :param who: which player, in (1, -1)
    :return nn_feature: converted tensor
    """
    assert who in (1, -1)
    env_torch = torch.tensor(env, dtype=torch.float32).unsqueeze(0)
    who_torch = torch.tensor(who, dtype=torch.int32)
    nn_feature = env_torch, who_torch
    return nn_feature


def get_nn_device(nn):
    """
    get the device of a nn
    :param nn:
    :return:
    """
    return next(nn.parameters()).device


def get_best_valid_move(p, valid_moves):
    """
    based on the nn output policy and the valid moves, calculate the best valid move
    :param p: prob vector, torch tensor
    :param valid_moves: a tuple of valid moves
    :return best_action: best action
    """
    assert len(valid_moves) > 0
    p = p.cpu().numpy().reshape(-1)
    best_action_p = 0.
    best_action = None
    for i, prob in enumerate(p):
        action = (i // 3, i % 3)
        if action in valid_moves and prob > best_action_p:
            best_action = action
            best_action_p = prob
    return best_action


class NNAgent(MCTSAgent.RandomAgent):
    """
    Agent plays TicTacToe guided by a neural network agent
    """
    def __init__(self, player, nn):
        super().__init__(player)
        import torch
        nn.eval()
        self.nn = nn
        nn.cuda()
        self.device = get_nn_device(nn)

    def assign_nn(self, nn):
        """
        assign a nn to the agent
        :param nn:
        """
        nn.eval()
        self.nn = nn
        nn.cuda()
        self.device = get_nn_device(nn)

    def policy(self, env):
        """
        Give action based on env it senses.
        :param env: TicTacToe environment, a 3*3 board
        :return action: the 2-tuple action
        """
        valid_moves = Env.get_valid_moves(env)
        nn_feature = convert_env_to_input(env, self.player)
        nn_feature = [item.to(self.device) for item in nn_feature]
        with torch.no_grad():
            p, _ = self.nn(nn_feature)
        a = get_best_valid_move(p, valid_moves)
        return a
