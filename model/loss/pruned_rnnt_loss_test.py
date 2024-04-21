# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.09
""" Unittest of PrunedRnntLoss """

import glog
import unittest
import torch

from model.loss.pruned_rnnt_loss import PrunedRnntLoss
from model.loss.pruned_rnnt_loss import PrunedRnntLossConfig

from model.joiner.joiner import Joiner, JoinerConfig


class TestPrunedRnntLoss(unittest.TestCase):
    """ Unittest of PrunedRnntLoss """

    def setUp(self) -> None:
        self._config = {"termination_symbol": 0, "reduction": "mean"}
        self._pruned_loss = PrunedRnntLoss(config=PrunedRnntLossConfig(
            **self._config))

        self._joiner_config = {
            "input_dim": 512,
            "output_dim": 1000,
            "activation": "relu",
            "prune_range": 5
        }
        self._joiner = Joiner(config=JoinerConfig(**self._joiner_config))

    def test_pruned_loss_forward(self):
        # Unittest of forward of PrunedRnntLoss.
        enc_out = torch.rand(4, 200, 512)
        enc_out_lengths = torch.Tensor([197, 200, 65, 80])
        pred_out = torch.rand(4, 16, 512)
        tgts_length = torch.Tensor([8, 10, 9, 2])
        tgts = torch.randint(1, 1000, (4, 15))

        logits, boundary, ranges, simple_loss = self._joiner(
            enc_out, enc_out_lengths, pred_out, tgts_length, tgts)

        pruned_loss = self._pruned_loss(logits, tgts, enc_out_lengths,
                                        tgts_length, boundary, ranges)
        glog.info("Pruned Rnnt Loss: {}".format(pruned_loss))
        glog.info("Simple Rnnt Loss: {}".format(simple_loss))


if __name__ == "__main__":
    unittest.main()
