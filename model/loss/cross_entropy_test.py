# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.11
""" Unittest of Cross-Entropy """

import glog
import torch
import unittest

from parameterized import parameterized
from model.loss.cross_entropy import MaskedCELoss, MaskedCELossConfig


class TestMaskedCELoss(unittest.TestCase):
    """ Unittest of MaskedCross-Entropy loss """

    def setUp(self) -> None:
        self._loss_config = {"num_classes": 8193}
        self._masked_ce_loss = MaskedCELoss(config=MaskedCELossConfig(
            **self._loss_config))

    @parameterized.expand([(723,), (671,), (437,)])
    def test_loss_forward(self, length):
        # Unittest of loss forward
        enc_out = torch.rand(8, length, 8193)
        enc_out_length = torch.tensor([20, 40, 100, 100, 100, 100, 100, length])
        ori_labels = torch.randint(0, self._loss_config["num_classes"],
                                   (8, length))
        loss = self._masked_ce_loss(enc_out, ori_labels, enc_out_length)
        glog.info("CE Loss: {}".format(loss))

    @parameterized.expand([(723,), (671,), (437,)])
    def test_predict(self, length):
        # Unittest of prediction
        logits = torch.rand(2, length, 8193)
        probs = self._masked_ce_loss.predict(logits=logits)
        self.assertEqual(probs.shape[-1], self._loss_config["num_classes"])
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(2,
                                                                     length)))


if __name__ == "__main__":
    unittest.main()
