import unittest
import Env
from Agents import MCTSAgent
import numpy as np
import copy


class TestMCTSAgent(unittest.TestCase):
    def test_expand(self):
        board = np.array([[0, 1, -1],
                          [0, -1, 1],
                          [0, 1, -1]])
        board_ = copy.deepcopy(board)
        game = Env.TicTacToe(board)
        node = MCTSAgent.MCTSNode(game, now_who=1, dirichlet_combining_epsilon=0)
        MCTSAgent.expand(node)
        # the current game state should not change after expanding
        self.assertTrue(np.allclose(node.game.board, board_))
        # 3 children
        self.assertEqual(len(node.children), 3)
        # stats should be init values
        for child in node.children:
            self.assertEqual(child.now_who, -1)
            self.assertEqual(child.parent, node)
            self.assertEqual(child.n, 0)
            self.assertEqual(child.w, 0)
            self.assertEqual(child.q, 0)
            self.assertAlmostEqual(child.p, 1/3)
            self.assertEqual(child.c_init, node.c_init)
            self.assertEqual(child.c_base, node.c_base)

        # children cases check
        self.assertTrue(np.allclose(node.children[0].game.board,
                        np.array([[ 1, 1,-1],
                                  [ 0,-1, 1],
                                  [ 0, 1,-1]])))
        self.assertEqual(node.children[0].a, (0, 0))
        self.assertTrue(np.allclose(node.children[1].game.board,
                        np.array([[0, 1, -1],
                                  [1, -1, 1],
                                  [0, 1, -1]])))
        self.assertEqual(node.children[1].a, (1, 0))
        self.assertTrue(np.allclose(node.children[2].game.board,
                        np.array([[0, 1, -1],
                                  [0, -1, 1],
                                  [1, 1, -1]])))
        self.assertEqual(node.children[2].a, (2, 0))

    def test_backprop(self):
        board = np.array([[0, 1, -1],
                          [0, -1, -1],
                          [0, 1, 1]])
        game = Env.TicTacToe(board)
        node = MCTSAgent.MCTSNode(game, None, -1)
        MCTSAgent.expand(node)
        MCTSAgent.expand(node.children[0])
        MCTSAgent.backprop(node.children[0].children[0], 1)
        self.assertEqual(node.children[0].n, 1)
        self.assertEqual(node.children[0].w, 1)
        self.assertEqual(node.children[0].q, 1)
        self.assertEqual(node.n, 1)
        self.assertEqual(node.w, 1)
        self.assertEqual(node.q, 1)

    def test_select_a_child(self):
        board = np.array([[0, 1, -1],
                          [0, -1, -1],
                          [0, 1, 1]])
        game = Env.TicTacToe(board)
        node = MCTSAgent.MCTSNode(game, None, 1)
        MCTSAgent.expand(node)
        MCTSAgent.expand(node.children[0])
        MCTSAgent.backprop(node.children[0].children[0], 1)
        self.assertEqual(MCTSAgent.select_a_child(node), node.children[0])
        self.assertEqual(MCTSAgent.select_a_child(node.children[0]), node.children[0].children[1])

    def test_selection(self):
        board = np.array([[0, 1, -1],
                          [0, -1, -1],
                          [0, 1, 1]])
        game = Env.TicTacToe(board)
        node = MCTSAgent.MCTSNode(game, None, 1)
        MCTSAgent.expand(node)
        MCTSAgent.expand(node.children[0])
        MCTSAgent.backprop(node.children[0].children[0], 1)
        self.assertEqual(MCTSAgent.selection(node), node.children[0].children[1])

        board = np.array([[0, 1, -1],
                          [0, -1, -1],
                          [0, 1, 1]])
        game = Env.TicTacToe(board)
        node = MCTSAgent.MCTSNode(game, None, -1)
        MCTSAgent.expand(node)
        MCTSAgent.expand(node.children[0])
        MCTSAgent.backprop(node.children[0].children[0], -1)
        self.assertEqual(MCTSAgent.selection(node), node.children[0].children[1])

    def test_simulation(self):
        board = np.array([[-1, 1, 1],
                          [1, -1, -1],
                          [0, -1, 1]])
        game = Env.TicTacToe(board)
        node = MCTSAgent.MCTSNode(game, None, 1)
        result = MCTSAgent.simulation(node)
        self.assertEqual(result, 0)

        board = np.array([[-1, 1, 1],
                          [1, -1, -1],
                          [0, 1, 1]])
        game = Env.TicTacToe(board)
        node = MCTSAgent.MCTSNode(game, None, 1)
        result = MCTSAgent.simulation(node)
        self.assertEqual(result, 1)

    def test_build_tree_choose_best_action(self):
        board = np.array([[-1, 1, 1],
                          [1, -1, -1],
                          [0, 1, 1]])
        game = Env.TicTacToe(board)
        node = MCTSAgent.MCTSNode(game)
        root = MCTSAgent.build_tree(node)
        a = MCTSAgent.choose_best_action(root)
        self.assertEqual(a, (2, 0))

        board = np.array([[0, -1, 1],
                          [0, -1, -1],
                          [0, 1, 1]])
        game = Env.TicTacToe(board)
        node = MCTSAgent.MCTSNode(game, now_who=-1)
        root = MCTSAgent.build_tree(node)
        a = MCTSAgent.choose_best_action(root)
        self.assertEqual(a, (1, 0))

        board = np.array([[0, -1, 1],
                          [0, -1, 1],
                          [0, 1, -1]])
        game = Env.TicTacToe(board)
        node = MCTSAgent.MCTSNode(game, now_who=1)
        root = MCTSAgent.build_tree(node)
        a = MCTSAgent.choose_best_action(root)
        self.assertEqual(a, (0, 0))

        board = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [-1, 1, -1]])
        game = Env.TicTacToe(board)
        node = MCTSAgent.MCTSNode(game, now_who=-1)
        root = MCTSAgent.build_tree(node)
        a = MCTSAgent.choose_best_action(root)
        self.assertEqual(a, (0, 1))

    def test_MCTSAgent(self):
        board = np.array([[1, 1, -1],
                          [0, -1, -1],
                          [0, 1, 1]])
        agent = MCTSAgent.MCTSAgent(1)
        a = agent.policy(board)
        self.assertEqual(a, (2, 0))

        board = np.array([[1, 1, -1],
                          [0, -1, -1],
                          [0, 1, 1]])
        agent = MCTSAgent.MCTSAgent(-1)
        a = agent.policy(board)
        self.assertEqual(a, (1, 0))

        board = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [-1, 1, -1]])
        agent = MCTSAgent.MCTSAgent(-1)
        a = agent.policy(board)
        self.assertEqual(a, (0, 1))


if __name__ == '__main__':
    unittest.main()
