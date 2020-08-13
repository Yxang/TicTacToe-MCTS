import multiprocessing
import Env
from Agents import RandomAgent, MCTSAgent
import logging
import queue

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class AgentProxy:
    """
    multiprocessing proxy for an agent, receiving the environments and sending the actions
    """
    def __init__(self, agent, action_q, env_q):
        """
        :param agent: the agent
        :param action_q: the action queue to send the actions
        :param env_q: the environment queue to get from the referee
        """
        self.agent = agent
        self.action_q = action_q
        self.env_q = env_q

    def evaluate(self):
        """
        play the game on env received, and send the action
        """
        env = self.env_q.get()
        a = self.agent.policy(env)
        self.action_q.put(a)


class GameProxy:
    """
    multiprocessing proxy for the game, receiving the actions from players, update the game, and sending the environment
    information

    agent 1 is the player with "X", which is 1,
    agent 2 is the player with "O", which is -1
    """
    def __init__(self, env_q_a1, env_q_a2, action_q_a1, action_q_a2, board=None):
        self.env_q = {1: env_q_a1,
                      -1: env_q_a2}
        self.action_q = {1: action_q_a1,
                         -1: action_q_a2}
        self.game = Env.TicTacToe(board)

    def sense(self, who):
        """
        send the env to the agent
        :param who: which agent, 1 is 1 or "X", -1 is 2 or "O"
        """
        assert who in (1, -1)
        env = self.game.get_env(who)
        self.env_q[who].put(env)

    def action(self, who):
        """
        perform the action received from player
        :param who: which agent, 1 is 1 or "X", -1 is 2 or "O"
        """
        assert who in (1, -1)
        a = self.action_q[who].get()
        game = self.game.action(a, who)
        self.game = game


def switch_turn(who):
    """
    switch the turn for player who, from 1 to -1 and from -1 to 1
    :param who:
    :return:
    """
    return -who


def agent_proxy(agent, action_q, env_q):
    """
    the function utilizes AgentProxy to used by multiprocessing Process
    :param agent: the agent type
    :param action_q: action info queue
    :param env_q: environment info queue
    """
    proxy = AgentProxy(agent, action_q, env_q)
    while True:
        # keep evaluating
        proxy.evaluate()


def game_proxy(env_q_a1, env_q_a2, action_q_a1, action_q_a2, result_q, start_who, log=False, board=None):
    """
    the function utilizes GameProxy to used by multiprocessing Process
    :param env_q_a1: environment info queue to agent 1
    :param env_q_a2: environment info queue to agent 2
    :param action_q_a1: action info queue to agent 1
    :param action_q_a2: action info queue to agent 2
    :param result_q: result queue to the referee
    :param start_who: start with player whom
    :param log: if logging
    :param board: start board. If None, start with empty board
    :return:
    """
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    proxy = GameProxy(env_q_a1, env_q_a2, action_q_a1, action_q_a2, board)
    status = proxy.game.check_game_state()
    who = start_who
    turn = 0
    while status is None:
        # let agent sense the env
        proxy.sense(who)
        # do the action
        proxy.action(who)
        # switch turn
        who = switch_turn(who)
        turn += 1
        # check status
        status = proxy.game.check_game_state()
        # log the game
        if log:
            logger.debug(f'Turn {turn}')
            logger.debug('Board: \n' + str(proxy.game))
    result_q.put(status)


class Referee:
    """
    The class that setup the processes of the agents and the game, and get the result.
    """
    def __init__(self):
        self.start_who = 1

        self.mt = False
        self.agent_proxy_p = dict()
        self.game_proxy_p = None
        self.to_agent1_env_q = None
        self.to_agent1_action_q = None
        self.to_agent2_env_q = None
        self.to_agent2_action_q = None
        self.result_q = None
        self.log = False

    def setup(self, agent1, agent2, log=False, board=None, mt=False):
        """
        setup the processes
        :param agent1: agent object for player 1, or "X", 1
        :param agent2: agent object for player 2, or "O", -1
        :param log: weather to log the game, passed to game_proxy
        :param board: the board to start with, passed to game_proxy
        :param mt: whether to use multiprocessing
        """
        self.mt = mt
        self.log = log
        if self.mt:
            self.to_agent1_env_q = multiprocessing.Queue()
            self.to_agent1_action_q = multiprocessing.Queue()
            self.to_agent2_env_q = multiprocessing.Queue()
            self.to_agent2_action_q = multiprocessing.Queue()
            self.result_q = multiprocessing.Queue()
            self.agent_proxy_p[1] = multiprocessing.Process(name='agent_1',
                                                            target=agent_proxy,
                                                            args=(agent1, self.to_agent1_action_q, self.to_agent1_env_q))
            self.agent_proxy_p[-1] = multiprocessing.Process(name='agent_2',
                                                             target=agent_proxy,
                                                             args=(agent2, self.to_agent2_action_q, self.to_agent2_env_q))

            self.game_proxy_p = multiprocessing.Process(name='game',
                                                        target=game_proxy,
                                                        args=(self.to_agent1_env_q,
                                                              self.to_agent2_env_q,
                                                              self.to_agent1_action_q,
                                                              self.to_agent2_action_q,
                                                              self.result_q,
                                                              self.start_who,
                                                              self.log,
                                                              board)
                                                        )
        else:
            self.to_agent1_env_q = queue.Queue()
            self.to_agent1_action_q = queue.Queue()
            self.to_agent2_env_q = queue.Queue()
            self.to_agent2_action_q = queue.Queue()
            self.result_q = queue.Queue()
            self.agent_proxy_p[1] = AgentProxy(agent1, self.to_agent1_action_q, self.to_agent1_env_q)
            self.agent_proxy_p[-1] = AgentProxy(agent2, self.to_agent2_action_q, self.to_agent2_env_q)
            self.game_proxy_p = GameProxy(self.to_agent1_env_q,
                                          self.to_agent2_env_q,
                                          self.to_agent1_action_q,
                                          self.to_agent2_action_q,
                                          board
                                          )

    def host(self):
        """
        host a whole game
        :return result: the result of the game
        """
        if self.mt:
            # multiprocessing version
            self.agent_proxy_p[1].start()
            self.agent_proxy_p[-1].start()
            self.game_proxy_p.start()
            result = self.result_q.get()
            self.agent_proxy_p[1].terminate()
            self.agent_proxy_p[-1].terminate()
            self.game_proxy_p.terminate()
        else:
            # single threaded version
            status = self.game_proxy_p.game.check_game_state()
            who = self.start_who
            turn = 0
            while status is None:
                self.game_proxy_p.sense(who)
                self.agent_proxy_p[who].evaluate()
                self.game_proxy_p.action(who)
                who = switch_turn(who)
                status = self.game_proxy_p.game.check_game_state()
                if self.log:
                    logger.debug(f'Turn {turn}')
                    logger.debug('Board: \n' + str(self.game_proxy_p.game))
                turn += 1
            result = status
        return result


if __name__ == '__main__':
    referee = Referee()
    agent1 = MCTSAgent.MCTSAgent(1)
    agent2 = MCTSAgent.MCTSAgent(-1)
    referee.setup(agent1, agent2, log=True, mt=True)
    result = referee.host()
    logger.debug(f'the result is {result}')
