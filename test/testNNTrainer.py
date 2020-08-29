import unittest
import torch
from torch.utils.data import DataLoader
import NNTrainer
import NNReferee
from Agents import NNAgent, MCTSNNAgent


class TestNNTrainer(unittest.TestCase):
    def test_nnDataset(self):
        net = NNAgent.NN()
        agent1 = {'agent': MCTSNNAgent.MCTSNNAgent, 'params': (1, net)}
        agent2 = {'agent': MCTSNNAgent.MCTSNNAgent, 'params': (-1, net)}
        referee = NNReferee.NNReferee()
        referee.setup(agent1, agent2, mt=False)
        result, training_data = referee.host()
        ds = NNTrainer.NNDataset(training_data)
        self.assertLessEqual(len(ds), 9)
        self.assertEqual(len(ds[0]), 3)
        dl = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
        for item in dl:
            self.assertEqual(len(item), 3)
            self.assertIsInstance(item[0], torch.Tensor)
            self.assertEqual(item[0].shape, (2, 2, 3, 3))
            self.assertIsInstance(item[1], torch.Tensor)
            self.assertEqual(item[1].shape, (2, 9))
            self.assertIsInstance(item[2], torch.Tensor)
            self.assertEqual(item[2].shape, (2, 1))
            break

        dl = DataLoader(ds, batch_size=1, collate_fn=ds.collate_fn)
        for item in dl:
            self.assertEqual(len(item), 3)
            self.assertIsInstance(item[0], torch.Tensor)
            self.assertEqual(item[0].shape, (1, 2, 3, 3))
            self.assertIsInstance(item[1], torch.Tensor)
            self.assertEqual(item[1].shape, (1, 9))
            self.assertIsInstance(item[2], torch.Tensor)
            self.assertEqual(item[2].shape, (1, 1))
            break

    def test_NNLoss(self):
        policy = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 1., 0., 0., 0., 0., 0., 0., 0.]])
        policy_hat = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 1., 0., 0., 0., 0., 0., 0., 0.]])
        value = torch.tensor([[1.], [-1.]])
        value_hat = torch.tensor([[1.], [-1.]])
        loss = NNTrainer.NNLoss()(policy, policy_hat, value, value_hat)
        self.assertEqual(loss.item(), 0.)

        policy = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 1., 0., 0., 0., 0., 0., 0., 0.]])
        policy_hat = torch.tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                   [1., 0., 0., 0., 0., 0., 0., 0., 0.]])
        value = torch.tensor([[1.], [-1.]])
        value_hat = torch.tensor([[1.], [-1.]])
        loss = NNTrainer.NNLoss()(policy, policy_hat, value, value_hat)
        self.assertAlmostEqual(loss.item(), 18.420681)

        policy = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 1., 0., 0., 0., 0., 0., 0., 0.]])
        policy_hat = torch.tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                   [1., 0., 0., 0., 0., 0., 0., 0., 0.]])
        value = torch.tensor([[1.], [-1.]])
        value_hat = torch.tensor([[-1.], [1.]])
        loss = NNTrainer.NNLoss()(policy, policy_hat, value, value_hat)
        self.assertAlmostEqual(loss.item(), 22.420681)

    def test_train(self):
        net = NNAgent.NN()
        agent1 = {'agent': MCTSNNAgent.MCTSNNAgent, 'params': (1, net)}
        agent2 = {'agent': MCTSNNAgent.MCTSNNAgent, 'params': (-1, net)}
        referee = NNReferee.NNReferee()
        referee.setup(agent1, agent2, mt=False)
        result, training_data = referee.host()
        trainer = NNTrainer.NNTrainer(net, training_data)
        trainer.train()
