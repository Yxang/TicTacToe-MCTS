import numpy as np
import Env

class RandomAgent:
    """
    Agent plays TicTacToe for randomly chosen moves
    """
    def __init__(self, player):
        self.set_player(player)

    def set_player(self, player):
        """
        set this agent as player 1 for "X" or -1 for "O"
        :param player: 1 for "X" or -1 for "O"
        """
        assert player in (1, -1)
        self.player = player

    def policy(self, env):
        """
        Give action based on env it senses.
        :param env: TicTacToe environment, a 3*3 board
        :return action: the 2-tuple action
        """
        valid_moves = Env.get_valid_moves(env)
        a = valid_moves[np.random.choice(len(valid_moves), 1).item()]
        return a
