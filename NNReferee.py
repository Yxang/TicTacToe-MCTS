import multiprocessing
import logging
import queue
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import Env
from Agents import RandomAgent, MCTSAgent, NNAgent, MCTSNNAgent
import Referee

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class NNAgentProxy(Referee.AgentProxy):
    """
    modified agent proxy, add the function of outputting the training data
    the training data is a (env, policy) pair
    """
    def __init__(self, agent, action_q, env_q, training_data_q):
        """
        :param agent: the agent config dict
        :param action_q: the action queue to send the actions
        :param env_q: the environment queue to get from the referee
        :param training_data_q: the queue to send training data for the nn
        """
        super().__init__(agent, action_q, env_q)
        self.training_data_q = training_data_q

    def send_training_data(self):
        """
        send the training data to training_data_q, where a training data is (nn_feature, policy)
        """
        assert isinstance(self.agent, MCTSNNAgent.MCTSNNAgent)
        self.training_data_q.put(self.agent.get_training_data())


def nn_agent_proxy(agent, action_q, env_q, training_data_q):
    """
    the function utilizes AgentProxy to used by multiprocessing Process
    :param agent: the agent config dict
    :param action_q: action info queue
    :param env_q: environment info queue
    :param training_data_q: the queue to send training data for the nn
    """
    proxy = NNAgentProxy(agent, action_q, env_q, training_data_q)
    while True:
        # keep evaluating
        try:
            proxy.evaluate()
            proxy.send_training_data()
        except Exception:
            traceback.print_exc()
            return


class NNReferee(Referee.Referee):
    def __init__(self):
        super().__init__()
        self.training_data_q = None

    def setup(self, agent1, agent2, log=False, board=None, start_who=1, mt=False):
        """
        setup the processes
        :param agent1: agent config dict for player 1, or "X", 1
        :param agent2: agent config dict player 2, or "O", -1
        :param log: weather to log the game, passed to game_proxy
        :param board: the board to start with, passed to game_proxy
        :param start_who: who's tern to start
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
            self.training_data_q = multiprocessing.Queue()
            self.agent_proxy_p[1] = multiprocessing.Process(name='agent_1',
                                                            target=nn_agent_proxy,
                                                            args=(agent1,
                                                                  self.to_agent1_action_q,
                                                                  self.to_agent1_env_q,
                                                                  self.training_data_q))
            self.agent_proxy_p[-1] = multiprocessing.Process(name='agent_2',
                                                             target=nn_agent_proxy,
                                                             args=(agent2,
                                                                   self.to_agent2_action_q,
                                                                   self.to_agent2_env_q,
                                                                   self.training_data_q))

            self.game_proxy_p = multiprocessing.Process(name='game',
                                                        target=Referee.game_proxy,
                                                        args=(self.to_agent1_env_q,
                                                              self.to_agent2_env_q,
                                                              self.to_agent1_action_q,
                                                              self.to_agent2_action_q,
                                                              self.result_q,
                                                              start_who,
                                                              self.log,
                                                              board))
        else:
            self.to_agent1_env_q = queue.Queue()
            self.to_agent1_action_q = queue.Queue()
            self.to_agent2_env_q = queue.Queue()
            self.to_agent2_action_q = queue.Queue()
            self.result_q = queue.Queue()
            self.training_data_q = queue.Queue()
            self.agent_proxy_p[1] = NNAgentProxy(agent1,
                                                 self.to_agent1_action_q,
                                                 self.to_agent1_env_q,
                                                 self.training_data_q)
            self.agent_proxy_p[-1] = NNAgentProxy(agent2,
                                                  self.to_agent2_action_q,
                                                  self.to_agent2_env_q,
                                                  self.training_data_q)
            self.game_proxy_p = Referee.GameProxy(self.to_agent1_env_q,
                                                  self.to_agent2_env_q,
                                                  self.to_agent1_action_q,
                                                  self.to_agent2_action_q,
                                                  board)

    def host(self):
        """
        host a whole game, manage the result, update the training data for NN
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
                self.agent_proxy_p[who].send_training_data()
                self.game_proxy_p.action(who)
                who = Referee.switch_turn(who)
                status = self.game_proxy_p.game.check_game_state()
                if self.log:
                    logger.debug(f'Turn {turn}')
                    logger.debug('Board: \n' + str(self.game_proxy_p.game))
                turn += 1
            result = status
        training_data = self.get_training_data(result)
        return result, training_data

    def get_training_data(self, result):
        """
        form the training data from the training_data_q
        :param result: the result of the game
        :return training_data: a list of (nn_feature, policy, value)
        """
        training_data_2 = []
        training_data = []
        while self.training_data_q.qsize() > 0:
            training_data_2.append(self.training_data_q.get())
        value = result
        value = torch.tensor(value)
        for nn_feature, policy in training_data_2:
            training_data.append((nn_feature, policy, value))
        return training_data
