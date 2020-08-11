import numpy as np
import copy

class TicTacToe:
    """
    TicTacToe game engine
    """

    def __init__(self, board=None):
        """
        Initialize the game state as all zeros (all empty board)

        :param board: the board init with
        """
        if board is None:
            self.board = np.zeros((3, 3))
        else:
            assert isinstance(board, np.ndarray)
            assert board.shape == (3,3)
            assert set(np.sort(np.unique(board))).issubset({-1, 0, 1})
            self.board = board
        self.state = None

    def get_env(self):
        return self.board

    def check_game_state(self):
        """
        check if the TicTacToe game has ended or not.
        If yes (game ended), return the game result (1: x_player win, -1: o_player win, 0: draw)
        If no (game not ended yet), return None

        :return e:
            the result, an integer scalar with value 0, 1 or -1.
            if e = None, the game has not ended yet.
            if e = 0, the game ended with a draw.
            if e = 1, X player won the game.
            if e = -1, O player won the game.
        """
        # check the 8 lines in the board to see if the game has ended.
        a = []
        a.extend(np.sum(self.board, axis=0).tolist())
        a.extend(np.sum(self.board, axis=1).tolist())
        a.append(np.sum([self.board[0, 0], self.board[1, 1], self.board[2, 2]]))
        a.append(np.sum([self.board[0, 2], self.board[1, 1], self.board[2, 0]]))
        # if the game has ended, return the game result
        if 3 in a:
            e = 1
        elif -3 in a:
            e = -1
        elif 0 in self.board:
            # if the game has not ended, return None
            e = None
        else:
            e = 0
        return e

    def __str__(self):
        map_dict = {0: '  ', 1: 'X ', -1: 'O '}
        r0 = '|'.join([map_dict[item] for item in self.board[0, :]])
        r1 = '|'.join([map_dict[item] for item in self.board[1, :]])
        r2 = '|'.join([map_dict[item] for item in self.board[2, :]])
        spliter = '\n--------\n'
        s = r0 + spliter + r1 + spliter + r2
        return s

    def action(self, a, who):
        """
        Do action a on the current game with player who. who=1 represent player X, and who=-1 represent player O
        :param a: the action, a 2-tuple represents the position of the next step
        :param who: who is making this action, should be 1 or -1
        :return g: game after action
        """
        assert self.check_game_state() is None  # check the game is not ended
        assert isinstance(a, tuple) and len(a) == 2  # check if the action is valid
        assert a[0] in (0, 1, 2) and a[1] in (0, 1, 2)  # check if the action is valid
        assert self.board[a] == 0  # check if the move is valid
        assert who in (1, -1)
        g = copy.deepcopy(self)
        g.board[a] = who
        return g

def get_valid_moves(env, player=None):
    """
    Get a list of available (valid) next moves from a game state of TicTacToe

    :param env: the current state of the game, an integer matrix of shape 3 by 3.
        env[i,j] = 0 denotes that the i-th row and j-th column is empty
        env[i,j] = 1 denotes that the i-th row and j-th column is taken by "X".
        env[i,j] = -1 denotes that the i-th row and j-th column is taken by the "O".
        For example, the following game state
         | X |   | O |
         | O | X |   |
         | X |   | O |
        is represented as the following numpy matrix in game state
        env= [[ 1 , 0 ,-1 ],
              [-1 , 1 , 0 ],
              [ 1 , 0 ,-1 ]]

    :return m: a list of possible next moves
        in which each next move is a (r,c) tuple,
        r denotes the row number, c denotes the column number.
        For example, for the following game state,
                s= [[ 1 , 0 ,-1 ],
                    [-1 , 1 , 0 ],
                    [ 1 , 0 ,-1 ]]
        the valid moves are the empty grid cells:
            (r=0,c=1) --- the first row, second column
            (r=1,c=2) --- the second row, the third column
            (r=2,c=1) --- the third row , the second column
        So the list of valid moves is m = [(0,1),(1,2),(2,1)]
    """
    m = [tuple(item) for item in np.argwhere(env == 0).tolist()]
    return m
