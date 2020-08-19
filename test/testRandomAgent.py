import unittest
from Agents import RandomAgent
import numpy as np


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


if __name__ == '__main__':
    unittest.main()
