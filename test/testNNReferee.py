import NNReferee
from Agents import NNAgent, MCTSNNAgent, RandomAgent
import Env
import numpy as np
import multiprocessing
import unittest
import torch


class TestNNReferee(unittest.TestCase):
    def test_nn_agent_proxy(self):
        net = NNAgent.NN()
        agent = {'agent': MCTSNNAgent.MCTSNNAgent, 'params': (1, net)}
        action_q = multiprocessing.Queue()
        env_q = multiprocessing.Queue()
        training_data_q = multiprocessing.Queue()
        p = multiprocessing.Process(target=NNReferee.nn_agent_proxy,
                                    args=(agent, action_q, env_q, training_data_q))
        p.start()

        board = np.array([[0, 1, 1],
                          [1, 0, -1],
                          [1, 1, 0]])
        env_q.put(board)
        action = action_q.get()
        self.assertTrue(action in Env.get_valid_moves(board))

        nn_feature, policy = training_data_q.get()
        self.assertTrue(isinstance(nn_feature, torch.Tensor), msg=f'nn_feature[0] is {type(nn_feature)}')
        self.assertTrue(nn_feature.shape == (1, 2, 3, 3))
        self.assertTrue(isinstance(policy, torch.Tensor))
        self.assertTrue(policy.shape == (9,))
        self.assertTrue(torch.all(policy[[0, 4, 8]] > 0))
        self.assertTrue(torch.all(policy[[1, 2, 3, 5, 6, 7]] == 0))

        p.terminate()

    def test_referee_host(self):
        net = NNAgent.NN()
        referee = NNReferee.NNReferee()
        for _ in range(2):
            agent1 = {'agent': MCTSNNAgent.MCTSNNAgent, 'params': (1, net)}
            agent2 = {'agent': MCTSNNAgent.MCTSNNAgent, 'params': (-1, net)}
            referee.setup(agent1, agent2, mt=True)
            result, training_data = referee.host()
            self.assertIn(result, (1, -1, 0))
            self.assertEqual(len(training_data), 9)
            self.assertEqual(len(training_data[0]), 3)
            nn_feature, policy, value = training_data[0]
            self.assertIsInstance(nn_feature, torch.Tensor)
            self.assertIsInstance(policy, torch.Tensor)
            self.assertIsInstance(value, torch.Tensor)
            self.assertEqual(nn_feature.shape, (1, 2, 3, 3), msg=f'the shape is {nn_feature[0].shape}')
            self.assertEqual(policy.shape, (9,), msg=f'the shape is {policy.shape}')
            self.assertEqual(value.item(), result, msg=f'the value is {value}')

        referee = NNReferee.NNReferee()
        for _ in range(2):
            agent1 = {'agent': MCTSNNAgent.MCTSNNAgent, 'params': (1, net)}
            agent2 = {'agent': MCTSNNAgent.MCTSNNAgent, 'params': (-1, net)}
            referee.setup(agent1, agent2, mt=False)
            result, training_data = referee.host()
            self.assertIn(result, (1, -1, 0))
            self.assertEqual(len(training_data), 9)
            self.assertEqual(len(training_data[0]), 3)
            nn_feature, policy, value = training_data[0]
            self.assertIsInstance(nn_feature, torch.Tensor)
            self.assertIsInstance(policy, torch.Tensor)
            self.assertIsInstance(value, torch.Tensor)
            self.assertEqual(nn_feature.shape, (1, 2, 3, 3), msg=f'the shape is {nn_feature[0].shape}')
            self.assertEqual(policy.shape, (9,), msg=f'the shape is {policy.shape}')
            self.assertEqual(value.item(), result, msg=f'the value is {value}')




