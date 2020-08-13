import unittest
import Env
import Referee
from Agents import RandomAgent, MCTSAgent
import multiprocessing
import numpy as np
import time
import copy


class TestGameEnv(unittest.TestCase):
    def test_get_valid_moves(self):
        s = np.array([[1, 0, -1],
                      [-1, 1, 0],
                      [1, 0, -1]])
        game = Env.TicTacToe(s)
        m = Env.get_valid_moves(game.get_env())
        self.assertTrue(type(m) == list)
        self.assertTrue(len(m) == 3)
        self.assertTrue(m[0] == (0, 1) or m[0] == (1, 2) or m[0] == (2, 1))
        self.assertTrue(m[1] == (0, 1) or m[1] == (1, 2) or m[1] == (2, 1))
        self.assertTrue(m[2] == (0, 1) or m[2] == (1, 2) or m[2] == (2, 1))
        self.assertTrue(m[0] != m[1] and m[1] != m[2])

        game = Env.TicTacToe()
        m = Env.get_valid_moves(game.get_env())
        self.assertTrue(len(m) == 9)

        s = np.array([[1, 0, -1],
                      [0, 0, 0],
                      [1, 0, -1]])
        game = Env.TicTacToe(s)
        m = Env.get_valid_moves(game.get_env())
        self.assertTrue(len(m) == 5)
        for i in m:
            self.assertTrue(i == (0, 1) or i == (1, 0) or i == (1, 1) or i == (1, 2) or i == (2, 1))

    def test_check_game_state(self):
        s = np.array([[1, 0, 1],
                      [0, 0, -1],
                      [0, -1, 0]])
        game = Env.TicTacToe(s)
        e = game.check_game_state()
        self.assertTrue(e is None)  # the game has not ended yet

        s = np.array([[1, -1, 1],
                      [0, 1, -1],
                      [-1, 1, -1]])
        game = Env.TicTacToe(s)
        e = game.check_game_state()
        self.assertTrue(e is None)  # the game has not ended yet

        s = np.array([[1, 1, 1],
                      [0, 0, -1],
                      [0, -1, 0]])
        game = Env.TicTacToe(s)
        e = game.check_game_state()
        self.assertTrue(e == 1)  # x player wins
        game = Env.TicTacToe(-s)
        e = game.check_game_state()
        self.assertTrue(e == -1)  # O player wins

        s = np.array([[-1, 0, 0],
                      [1, 1, 1],
                      [0, -1, 0]])
        game = Env.TicTacToe(s)
        e = game.check_game_state()
        self.assertTrue(e == 1)  # x player wins
        game = Env.TicTacToe(-s)
        e = game.check_game_state()
        self.assertTrue(e == -1)  # O player wins

        s = np.array([[-1, 0, 0],
                      [0, 0, -1],
                      [1, 1, 1]])
        game = Env.TicTacToe(s)
        e = game.check_game_state()
        self.assertTrue(e == 1)  # x player wins
        game = Env.TicTacToe(-s)
        e = game.check_game_state()
        self.assertTrue(e == -1)  # O player wins

        s = np.array([[1, 0, 0],
                      [1, 0, -1],
                      [1, -1, 0]])
        game = Env.TicTacToe(s)
        e = game.check_game_state()
        self.assertTrue(e == 1)  # x player wins
        game = Env.TicTacToe(-s)
        e = game.check_game_state()
        self.assertTrue(e == -1)  # O player wins

        s = np.array([[-1, 1, 0],
                      [0, 1, 0],
                      [-1, 1, 0]])
        game = Env.TicTacToe(s)
        e = game.check_game_state()
        self.assertTrue(e == 1)  # x player wins
        game = Env.TicTacToe(-s)
        e = game.check_game_state()
        self.assertTrue(e == -1)  # O player wins

        s = np.array([[-1, 0, 1],
                      [0, 0, 1],
                      [-1, 0, 1]])
        game = Env.TicTacToe(s)
        e = game.check_game_state()
        self.assertTrue(e == 1)  # x player wins
        game = Env.TicTacToe(-s)
        e = game.check_game_state()
        self.assertTrue(e == -1)  # O player wins

        s = np.array([[1, 0, 0],
                      [0, 1, -1],
                      [-1, 0, 1]])
        game = Env.TicTacToe(s)
        e = game.check_game_state()
        self.assertTrue(e == 1)  # x player wins
        game = Env.TicTacToe(-s)
        e = game.check_game_state()
        self.assertTrue(e == -1)  # O player wins

        s = np.array([[-1, 0, 1],
                      [0, 1, 0],
                      [1, 0, -1]])
        game = Env.TicTacToe(s)
        e = game.check_game_state()
        self.assertTrue(e == 1)  # x player wins
        game = Env.TicTacToe(-s)
        e = game.check_game_state()
        self.assertTrue(e == -1)  # O player wins

        s = np.array([[-1, 1, -1],
                      [1, 1, -1],
                      [1, -1, 1]])
        game = Env.TicTacToe(s)
        e = game.check_game_state()
        self.assertTrue(e == 0)  # a tiegit

    def test_action(self):
        s = np.array([[1, 1, 1],
                      [0, 0, -1],
                      [0, -1, 0]])
        game = Env.TicTacToe(s)
        with self.assertRaises(AssertionError):
            game.action((1, 0), 1)

        s = np.array([[1, -1, 1],
                      [-1, 1, -1],
                      [1, -1, -1]])
        game = Env.TicTacToe(s)
        with self.assertRaises(AssertionError):
            game.action((1, 0), 1)

        s = np.array([[1, 0, 1],
                      [0, 0, -1],
                      [0, -1, 0]])
        game = Env.TicTacToe(s)
        with self.assertRaises(AssertionError):
            game.action((0, 1), 2)
        with self.assertRaises(AssertionError):
            game.action((0, 2), 1)
        with self.assertRaises(AssertionError):
            game.action(0, 1)
        with self.assertRaises(AssertionError):
            game.action((0, 3), 2)

        after_game = game.action((0, 1), 1)
        self.assertTrue(np.alltrue(after_game.board == np.array([[1, 1, 1],
                                                                 [0, 0, -1],
                                                                 [0, -1, 0]])))


class TestRandomAgent(unittest.TestCase):
    def test_set_player(self):
        agent = RandomAgent.RandomAgent(1)

        agent.set_player(1)
        self.assertEqual(agent.player, 1)

        agent.set_player(-1)
        self.assertEqual(agent.player, -1)

        with self.assertRaises(AssertionError):
            agent.set_player('1234')

        with self.assertRaises(AssertionError):
            agent.set_player(0)

    def test_policy(self):
        agent = RandomAgent.RandomAgent(1)
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
        self.assertGreater(count[0, 0], 20)
        self.assertGreater(count[1, 1], 20)
        self.assertGreater(count[2, 2], 20)


class TestReferee(unittest.TestCase):
    def test_switch_turn(self):
        self.assertEqual(-1, Referee.switch_turn(1))
        self.assertEqual(1, Referee.switch_turn(-1))

    def test_agent_proxy(self):
        agent = RandomAgent.RandomAgent(1)
        action_q = multiprocessing.Queue()
        env_q = multiprocessing.Queue()
        p = multiprocessing.Process(target=Referee.agent_proxy,
                                    args=(agent, action_q, env_q))
        p.start()

        board = np.array([[ 0, 1, 1],
                          [ 1, 0,-1],
                          [ 1, 1, 0]])
        env_q.put(board)
        action = action_q.get()
        self.assertTrue(action in Env.get_valid_moves(board))

        p.terminate()

    def test_game_proxy(self):
        env_q_a1 = multiprocessing.Queue()
        env_q_a2 = multiprocessing.Queue()
        action_q_a1 = multiprocessing.Queue()
        action_q_a2 = multiprocessing.Queue()
        result_q = multiprocessing.Queue()

        p = multiprocessing.Process(target=Referee.game_proxy,
                                    args=(env_q_a1, env_q_a2, action_q_a1, action_q_a2, result_q, 1, False))
        p.start()

        time.sleep(0.5)
        self.assertTrue(not env_q_a1.empty())
        self.assertTrue(env_q_a2.empty())
        env_q_a1.get()
        action_q_a1.put((0, 0))
        env_q_a2.get()
        action_q_a2.put((0, 1))
        env_q_a1.get()
        action_q_a1.put((0, 2))
        env_q_a2.get()
        action_q_a2.put((1, 0))
        env_q_a1.get()
        action_q_a1.put((1, 1))
        env_q_a2.get()
        action_q_a2.put((1, 2))
        self.assertTrue(result_q.empty())
        env_q_a1.get()
        action_q_a1.put((2, 0))
        time.sleep(0.1)
        self.assertFalse(result_q.empty())
        result = result_q.get()
        self.assertEqual(result, 1)
        self.assertFalse(p.is_alive())

    def test_referee(self):
        referee = Referee.Referee()
        for _ in range(5):
            agent1 = RandomAgent.RandomAgent(1)
            agent2 = RandomAgent.RandomAgent(-1)
            referee.setup(agent1, agent2, mt=True)
            result = referee.host()
            self.assertIn(result, (1, -1, 0))
        referee = Referee.Referee()
        for _ in range(5):
            agent1 = RandomAgent.RandomAgent(1)
            agent2 = RandomAgent.RandomAgent(-1)
            referee.setup(agent1, agent2, mt=False)
            result = referee.host()
            self.assertIn(result, (1, -1, 0))


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
