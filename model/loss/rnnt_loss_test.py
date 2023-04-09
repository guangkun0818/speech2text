# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.09
""" Unittest of Rnnt Loss """

import glog
import unittest
import torch

from model.loss.rnnt_loss import RnntLossConfig, RnntLoss


class TestRnntLoss(unittest.TestCase):

    def setUp(self) -> None:
        self._config = {"blank_label": 0, "clamp": -1, "reduction": "mean"}
        self._rnnt_loss = RnntLoss(config=RnntLossConfig(**self._config))

    def test_rnnt_loss_forward(self):
        # Unittest of Rnt Loss forward
        logits = torch.rand(2, 200, 10, 6)  # (B, T, 1 + U, D)
        targets = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                [2, 2, 1, 1, 1, 8, 6, 0,
                                 0]]).to(torch.int32)  # (B, U)
        logit_lengths = torch.Tensor([200, 160]).to(torch.int32)  # (B)
        target_lengths = torch.Tensor([9, 7]).to(torch.int32)  # (B)

        loss = self._rnnt_loss(logits, targets, logit_lengths, target_lengths)
        glog.info("Rnnt Loss: {}".format(loss))


if __name__ == "__main__":
    unittest.main()
