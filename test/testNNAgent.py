import unittest
import Env
from Agents import NNAgent
import numpy as np


class TestNNAgent(unittest.TestCase):
    def test_convert_env_to_input(self):
        import torch
        board = np.array([[0, 1, 1],
                          [1, 0, -1],
                          [1, 1, 0]])
        with self.assertRaises(AssertionError):
            NNAgent.convert_env_to_input(board, 2)
        t = NNAgent.convert_env_to_input(board, 1)
        self.assertTrue(isinstance(t, torch.Tensor))
        self.assertTrue(np.allclose(t.shape, [1, 2, 3, 3]))
        b0 = t[0, 0, :].numpy()
        self.assertTrue(np.allclose(b0,
                                    np.array([[0, 1, 1],
                                              [1, 0, 0],
                                              [1, 1, 0]])))
        b1 = t[0, 1, :].numpy()
        self.assertTrue(np.allclose(b1,
                                    np.array([[0, 0, 0],
                                              [0, 0, 1],
                                              [0, 0, 0]])))

    def test_get_best_valid_move(self):
        import torch
        p = torch.tensor([.1, .1, .1, .1, .1, .1, .15, .05, .1])
        board = np.array([[1, 1, 0],
                          [0, 0, 0],
                          [0, 0, 0]])
        valid_moves = Env.get_valid_moves(board)
        a = NNAgent.get_best_valid_move(p, valid_moves)
        self.assertTrue(a, (2, 0))

        p.cuda()
        a = NNAgent.get_best_valid_move(p, valid_moves)
        self.assertTrue(a, (2, 0))

    def test_policy(self):
        net = NNAgent.NN()
        agent = NNAgent.NNAgent(1, net)
        env = np.array([[0, 1, 1],
                        [1, 0, -1],
                        [1, 1, 0]])

        env_ = np.array([[0, 1, 1],
                         [1, 0, -1],
                         [1, 1, 0]])
        count = {(0, 0): 0,
                 (1, 1): 0,
                 (2, 2): 0}
        for _ in range(100):
            a = agent.policy(env)
            self.assertEqual(env[a], 0)
            self.assertEqual(a[0], a[1])
            self.assertTrue(a[0] in (0, 1, 2))
            self.assertTrue(np.allclose(env, env_))
            count[a] += 1

        self.assertEqual(max(count.values()), 100)


if __name__ == '__main__':
    unittest.main()