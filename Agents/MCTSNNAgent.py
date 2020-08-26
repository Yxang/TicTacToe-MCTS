import numpy as np
import Env
from Agents.NNAgent import NNAgent, convert_env_to_input
import Referee
import torch

from Agents.MCTSAgent import *


class MCTSNNNode(MCTSNode):
    """
    MCTS node for neural networks
    """
    def get_policy(self):
        """
        get the policy vector from this node, which is the visit frequency
        :return probs: visit frequency, a numpy array of length 9
        """
        probs_dict = {child.a: child.n / self.n for child in self.children}
        probs = []
        for i in range(9):
            a = (i // 3, i % 3)
            prob = probs_dict.get(a)
            if prob is None:
                prob = 0.
            probs.append(prob)
        probs = np.array(probs)
        return probs

    def get_training_data(self):
        """
        get the training data for this node, including the env and the visit frequency (policy)
        :return env: the env of this node, a numpy array
        :return policy: the policy vector, the visit frequency, a numpy array of length 9
        """
        env = self.game.get_env()
        policy = self.get_policy()
        return env, policy


def simulation(node, nn):
    """
    simulation: use a neural network agent to simulate the game to a terminal state
    :param node: node to simulate
    :param nn: the neural network passed to the agent
    :return result: the simulation result
    """
    referee = Referee.Referee()
    agent1 = {'agent': NNAgent, 'params': (1, nn)}
    agent2 = {'agent': NNAgent, 'params': (-1, nn)}
    referee.setup(agent1, agent2, board=node.game.board, start_who=node.now_who)
    result = referee.host()
    v = result
    return v


def build_tree(root, nn, n_sim=1000):
    """
    given a root node, build a monte-carlo search tree.
    Selection -> expand -> simulation -> backprop
    :param root: root node
    :param nn: the neural network
    :param n_sim: numbers for the simulation
    :return root: root node
    """
    for i in range(n_sim):
        # selection
        leaf = selection(root)

        # expand
        # check if leaf node is terminal state
        # if not true, expand it
        if leaf.game.check_game_state() is None:
            leaf = expand(leaf)

        # simulation
        v = simulation(leaf, nn)
        # backprop
        backprop(leaf, v)
    return root


class MCTSNNAgent(MCTSAgent):
    """
    Agent plays TicTacToe using MCTS policy, and report the training data for the neural network
    """

    def __init__(self, player, nn, n_sim=1000, mt=False, c_base=.1, c_init=1):
        super().__init__(player, n_sim, mt, c_base, c_init)
        self.nn = nn

    def policy(self, env):
        """
        Give action based on env it senses.
        :param env: TicTacToe environment, a 3*3 board
        :return action: the 2-tuple action
        """
        game = Env.TicTacToe(env)
        root = MCTSNNNode(game, now_who=self.player, c_base=self.c_base, c_init=self.c_init)
        self.root = root
        build_tree(root, self.nn, self.n_sim)
        a = choose_best_action(root)
        return a

    def get_training_data(self):
        """
        get the training data for this agent of the root node
        :return nn_feature: the input to the neural network, a tuple of torch tensor.
                            The first element is the env, the second is the player (who)
        :return policy: the policy vector, the visit frequency, converted to torch tensor
        """
        assert isinstance(self.root, MCTSNNNode)
        env, policy = self.root.get_training_data()
        nn_feature = convert_env_to_input(env, self.player)
        policy = torch.tensor(policy)
        return nn_feature, policy
