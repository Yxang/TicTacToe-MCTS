from Agents import RandomAgent
import Env
import numpy as np


class HumanAgent(RandomAgent.RandomAgent):
    """
    Agent plays TicTacToe that accept human input
    """

    def _input_to_action(self, string):
        a = string.strip().split(',')
        if len(a) != 2:
            return 'wrong move'
        a = tuple([int(item) for item in a])
        return a

    def policy(self, env):
        """
        Give action based on env it senses.
        :param env: TicTacToe environment, a 3*3 board
        :return action: the 2-tuple action
        """
        valid_moves = Env.get_valid_moves(env)
        print(f'Player {self.player}\'s turn')
        print('The board is')
        print(Env.TicTacToe(env))
        a = self._input_to_action(input('input your action in format "row, column":\n'))
        while a == 'wrong move' or a not in valid_moves:
            a = self._input_to_action(input('not valid move!\ninput your action in format row, column:\n'))
        return a


if __name__ == '__main__':
    env = np.array([[0, 1, 1],
                    [1, 0, -1],
                    [1, 1, 0]])
    agent = HumanAgent(1)
    agent.policy(env)
