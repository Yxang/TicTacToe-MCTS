import unittest
import Env
import numpy as np


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


if __name__ == '__main__':
    unittest.main()
