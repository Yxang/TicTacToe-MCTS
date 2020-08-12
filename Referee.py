import multiprocessing
import Env
import RandomAgent
import logging
import time

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
        :param a: the action
        :param who: which agent, 1 is 1 or "X", -1 is 2 or "O"
        """
        assert who in (1, -1)
        a = self.action_q[who].get()
        game = self.game.action(a, who)
        self.game = game


def switch_turn(who):
    return -who


def agent_proxy(agent, action_q, env_q):
    proxy = AgentProxy(agent, action_q, env_q)
    while True:
        # keep evaluating
        proxy.evaluate()


def game_proxy(env_q_a1, env_q_a2, action_q_a1, action_q_a2, result_q, start_who, board=None):
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
        logger.debug(f'Turn {turn}')
        logger.debug('Board: \n' + str(proxy.game))
    result_q.put(status)


class Referee:
    def __init__(self):
        self.start_who = 1

        self.agent_proxy_p = dict()
        self.game_proxy_p = None
        self.to_agent1_env_q = multiprocessing.Queue()
        self.to_agent1_action_q = multiprocessing.Queue()
        self.to_agent2_env_q = multiprocessing.Queue()
        self.to_agent2_action_q = multiprocessing.Queue()
        self.result_q = multiprocessing.Queue()

    def setup(self, agent1, agent2):
        """
        setup the processes
        :param agent1: agent object for player 1, or "X", 1
        :param agent2: agent object for player 2, or "O", -1
        """
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
                                                          self.start_who)
                                                    )

    def host(self):
        """
        host a whole game
        :return result: the result of the game
        """
        self.agent_proxy_p[1].start()
        self.agent_proxy_p[-1].start()
        self.game_proxy_p.start()
        result = self.result_q.get()
        self.agent_proxy_p[1].terminate()
        self.agent_proxy_p[-1].terminate()
        self.game_proxy_p.terminate()
        return result


if __name__ == '__main__':
    referee = Referee()
    agent1 = RandomAgent.RandomAgent()
    agent2 = RandomAgent.RandomAgent()
    referee.setup(agent1, agent2)
    result = referee.host()
    logger.debug(f'the result is {result}')
