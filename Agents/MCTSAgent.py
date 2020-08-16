import numpy as np
import Env
from Agents.RandomAgent import RandomAgent
import Referee


class MCTSNode:
    """
    Monte-Carlo Tree Search node
    """
    def __init__(self, game, a=None, now_who=1, parent=None, children=None, n=0, w=0, q=0, p=1.,
                 c_base=1., c_init=1., dirichlet_alpha=0.03, dirichlet_combining_epsilon=0.1):
        """
        :param game: the current game, consists the board info
        :param a: last action
        :param now_who: now who's turn
        :param parent: parent node
        :param children: children node list
        :param n: visit count
        :param w: total action-value, +1 for win, -1 for lose, 0 for draw
        :param q: mean action-value, w/n
        :param p: prior probability
        :param c_base: c_base in the PUCT upper bound
        :param c_init: c_init in the PUCT upper bound
        """
        self.game = game
        self.a = a
        self.now_who = now_who
        self.parent = parent
        self.children = children if children is not None else []
        self.n = n
        self.w = w
        self.q = q
        self.p = p
        self.c_base = c_base
        self.c_init = c_init
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_combining_epsilon = dirichlet_combining_epsilon


def expand(node):
    """
    expand a node to all its possible actions
    :param node: node to expand
    :return child: a random (the first) child of the expanded node
    """
    actions = Env.get_valid_moves(node.game.board)
    next_who = Referee.switch_turn(node.now_who)
    p = 1 / len(actions)
    if node.parent is None:
        dirichlet = np.random.dirichlet([node.dirichlet_alpha])
        p = p * (1 - node.dirichlet_combining_epsilon) + dirichlet * node.dirichlet_combining_epsilon
    for action in actions:
        next_game = node.game.action(action, who=node.now_who)
        node.children.append(MCTSNode(game=next_game, a=action, now_who=next_who, parent=node, p=p))
    child = node.children[0]
    return child


def backprop(node, v):
    """
    back propagation after simulation
    :param node: node to backprop
    :param v: value gained
    """
    this_node = node
    while this_node is not None:
        # change stats
        this_node.n += 1
        this_node.w += v
        this_node.q = this_node.w / this_node.n
        # change pointer
        this_node = this_node.parent


def compute_PUCT(node, who):
    """
    compute the PUCT algorithm stated in the paper
    :param node: node to compute. A PUCT is a f(s, a).
    :return score: the score to argmax
    """
    n = node.parent.n
    c = np.log((1 + n + node.c_base) / node.c_base) + node.c_init
    u = c * node.p * np.sqrt(n) / (1 + node.n)
    #u = node.c_init * np.sqrt(n / (1 + node.n))
    score = who * node.q + u
    return score


def select_a_child(node):
    """
    select a child of the node with highest PUCT score
    :param node: node to select
    :return child: the child with highest PUCT score
    """
    puct_scores = [compute_PUCT(n, node.now_who) for n in node.children]
    best = np.argmax(puct_scores)
    child = node.children[best]
    return child


def selection(node):
    """
    selection of the node all down to a leaf node
    :param node: node to do the selection
    :return leaf: the selected leaf node based on max PUCT score
    """
    this_node = node
    while len(this_node.children) > 0:
        this_node = select_a_child(this_node)
    return this_node


def simulation(node):
    """
    simulation: use random agent to simulate the game to a terminal state
    :param node: node to simulate
    :return result: the simulation result
    """
    referee = Referee.Referee()
    agent1 = {'agent': RandomAgent, 'params': (1,)}
    agent2 = {'agent': RandomAgent, 'params': (1,)}
    referee.setup(agent1, agent2, board=node.game.board)
    result = referee.host()
    v = result
    return v


def build_tree(root, n_sim=1000):
    """
    given a root node, build a monte-carlo search tree.
    Selection -> expand -> simulation -> backprop
    :param root: root node
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
        v = simulation(leaf)
        # backprop
        backprop(leaf, v)
    return root


def choose_best_action(root):
    """
    choose the best action based on the a expanded root node
    :param root: the root of the MCTS
    :return action: the optimal action
    """
    ns = [node.n for node in root.children]
    best = np.argmax(ns).item()
    action = root.children[best].a
    return action


class MCTSAgent(RandomAgent):
    """
    Agent plays TicTacToe using MCTS policy
    """
    def __init__(self, player, n_sim=1000, mt=False, c_base=.1, c_init=1):
        super(MCTSAgent, self).__init__(player)
        self.n_sim = n_sim
        self.mt = mt
        self.c_base = c_base
        self.c_init = c_init

    def policy(self, env):
        """
        Give action based on env it senses.
        :param env: TicTacToe environment, a 3*3 board
        :return action: the 2-tuple action
        """
        game = Env.TicTacToe(env)
        root = MCTSNode(game, now_who=self.player, c_base=self.c_base, c_init=self.c_init)
        build_tree(root, self.n_sim)
        a = choose_best_action(root)
        return a
