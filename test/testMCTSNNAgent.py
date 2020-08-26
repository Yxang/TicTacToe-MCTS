import numpy as np
import Env
from Agents import MCTSAgent, MCTSNNAgent, NNAgent
import unittest


class TestMCTSNNAgent(unittest.TestCase):
    def test_NNMCTSNode_get_policy(self):
        board = np.array([[0, 1, -1],
                          [0, -1, -1],
                          [0, 1, 1]])
        game = Env.TicTacToe(board)
        node = MCTSNNAgent.MCTSNNNode(game, None, 1)
        root = MCTSAgent.build_tree(node)
        policy = root.get_policy()
        self.assertEqual(np.argmax(policy), 6)

        board = np.array([[0, -1, 1],
                          [0, -1, 1],
                          [0, 1, -1]])
        game = Env.TicTacToe(board)
        node = MCTSNNAgent.MCTSNNNode(game, None, 1)
        root = MCTSAgent.build_tree(node)
        policy = root.get_policy()
        self.assertEqual(np.argmax(policy), 0)

        board = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [-1, 1, -1]])
        game = Env.TicTacToe(board)
        node = MCTSNNAgent.MCTSNNNode(game, None, -1)
        root = MCTSAgent.build_tree(node)
        policy = root.get_policy()
        self.assertEqual(np.argmax(policy), 1)

    def test_simulation(self):
        nn = NNAgent.NN()
        board = np.array([[-1, 1, 1],
                          [1, -1, -1],
                          [0, -1, 1]])
        game = Env.TicTacToe(board)
        node = MCTSNNAgent.MCTSNNNode(game, None, 1)
        result = MCTSNNAgent.simulation(node, nn)
        self.assertEqual(result, 0)

        board = np.array([[-1, 1, 1],
                          [1, -1, -1],
                          [0, 1, 1]])
        game = Env.TicTacToe(board)
        node = MCTSNNAgent.MCTSNNNode(game, None, 1)
        result = MCTSNNAgent.simulation(node, nn)
        self.assertEqual(result, 1)

    def test_build_tree(self):
        nn = NNAgent.NN()

        board = np.array([[-1, 1, 1],
                          [1, -1, -1],
                          [0, 1, 1]])
        game = Env.TicTacToe(board)
        node = MCTSNNAgent.MCTSNNNode(game)
        root = MCTSNNAgent.build_tree(node, nn)
        a = MCTSNNAgent.choose_best_action(root)
        self.assertEqual(a, (2, 0))

        board = np.array([[0, -1, 1],
                          [0, -1, -1],
                          [0, 1, 1]])
        game = Env.TicTacToe(board)
        node = MCTSNNAgent.MCTSNNNode(game, now_who=-1)
        root = MCTSNNAgent.build_tree(node, nn)
        a = MCTSNNAgent.choose_best_action(root)
        self.assertEqual(a, (1, 0))

        board = np.array([[0, -1, 1],
                          [0, -1, 1],
                          [0, 1, -1]])
        game = Env.TicTacToe(board)
        node = MCTSNNAgent.MCTSNNNode(game, now_who=1)
        root = MCTSNNAgent.build_tree(node, nn)
        a = MCTSNNAgent.choose_best_action(root)
        self.assertEqual(a, (0, 0))

        board = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [-1, 1, -1]])
        game = Env.TicTacToe(board)
        node = MCTSNNAgent.MCTSNNNode(game, now_who=-1)
        root = MCTSNNAgent.build_tree(node, nn)
        a = MCTSNNAgent.choose_best_action(root)
        self.assertEqual(a, (0, 1))

    def test_get_training_data(self):
        nn = NNAgent.NN()

        board = np.array([[-1, 1, 1],
                          [1, -1, -1],
                          [0, 1, 1]])
        game = Env.TicTacToe(board)
        node = MCTSNNAgent.MCTSNNNode(game)
        root = MCTSNNAgent.build_tree(node, nn)
        env, policy = root.get_training_data()
        self.assertTrue(np.allclose(env, board))
        self.assertEqual(np.argmax(policy), 6)

        board = np.array([[0, -1, 1],
                          [0, -1, -1],
                          [0, 1, 1]])
        game = Env.TicTacToe(board)
        node = MCTSNNAgent.MCTSNNNode(game, now_who=-1)
        root = MCTSNNAgent.build_tree(node, nn)
        env, policy = root.get_training_data()
        self.assertTrue(np.allclose(env, board))
        self.assertEqual(np.argmax(policy), 3)

        board = np.array([[0, -1, 1],
                          [0, -1, 1],
                          [0, 1, -1]])
        game = Env.TicTacToe(board)
        node = MCTSNNAgent.MCTSNNNode(game, now_who=1)
        root = MCTSNNAgent.build_tree(node, nn)
        env, policy = root.get_training_data()
        self.assertTrue(np.allclose(env, board))
        self.assertEqual(np.argmax(policy), 0)

        board = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [-1, 1, -1]])
        game = Env.TicTacToe(board)
        node = MCTSNNAgent.MCTSNNNode(game, now_who=-1)
        root = MCTSNNAgent.build_tree(node, nn)
        env, policy = root.get_training_data()
        self.assertTrue(np.allclose(env, board))
        self.assertEqual(np.argmax(policy), 1)

    def test_MCTSNNAgent(self):
        net = NNAgent.NN()
        agent = MCTSNNAgent.MCTSNNAgent(1, net)

        board = np.array([[1, 1, -1],
                          [0, -1, -1],
                          [0, 1, 1]])
        a = agent.policy(board)
        self.assertEqual(a, (2, 0))

        board = np.array([[1, 1, -1],
                          [0, -1, -1],
                          [0, 1, 1]])
        agent = MCTSNNAgent.MCTSNNAgent(-1, net)
        a = agent.policy(board)
        self.assertEqual(a, (1, 0))

        board = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [-1, 1, -1]])
        agent = MCTSNNAgent.MCTSNNAgent(-1, net)
        a = agent.policy(board)
        self.assertEqual(a, (0, 1))


if __name__ == '__main__':
    unittest.main()
