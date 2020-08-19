import unittest
import Env
import Referee
from Agents import RandomAgent
import multiprocessing
import numpy as np
import time


class TestReferee(unittest.TestCase):
    def test_switch_turn(self):
        self.assertEqual(-1, Referee.switch_turn(1))
        self.assertEqual(1, Referee.switch_turn(-1))

    def test_game_proxy(self):
        env_q_a1 = multiprocessing.Queue()
        env_q_a2 = multiprocessing.Queue()
        action_q_a1 = multiprocessing.Queue()
        action_q_a2 = multiprocessing.Queue()
        result_q = multiprocessing.Queue()

        p = multiprocessing.Process(name='game',
                                    target=Referee.game_proxy,
                                    args=(env_q_a1,
                                          env_q_a2,
                                          action_q_a1,
                                          action_q_a2,
                                          result_q,
                                          1,
                                          False)
                                    )
        p.start()
        self.assertTrue(np.allclose(env_q_a1.get(), np.zeros((3, 3))))
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
        result = result_q.get()
        self.assertEqual(result, 1)
        time.sleep(.5)
        self.assertFalse(p.is_alive())

    def test_agent_proxy(self):
        agent = {'agent': RandomAgent.RandomAgent, 'params': (1,)}
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

    def test_referee(self):
        referee = Referee.Referee()
        for _ in range(1):
            agent1 = {'agent': RandomAgent.RandomAgent, 'params': (1,)}
            agent2 = {'agent': RandomAgent.RandomAgent, 'params': (-1,)}
            referee.setup(agent1, agent2, mt=True)
            result = referee.host()
            self.assertIn(result, (1, -1, 0))
        referee = Referee.Referee()
        for _ in range(1):
            agent1 = {'agent': RandomAgent.RandomAgent, 'params': (1,)}
            agent2 = {'agent': RandomAgent.RandomAgent, 'params': (-1,)}
            referee.setup(agent1, agent2, mt=False)
            result = referee.host()
            self.assertIn(result, (1, -1, 0))

if __name__ == '__main__':
    unittest.main()
