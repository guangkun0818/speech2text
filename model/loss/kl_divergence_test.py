# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.11
""" Unittest of MaskedKLDivergence """

import glog
import torch
import unittest

from parameterized import parameterized
from model.loss.kl_divergence import MaskedKLDivergence, MaskedKLDivergenceConfig


class TestMaskedKLDivergence(unittest.TestCase):

    def setUp(self) -> None:
        self._loss_config = {
            "num_classes": 8193,
            "scale_factor": 1.0,
            "label_smoothing": 0.1
        }
        self._kl_loss = MaskedKLDivergence(config=MaskedKLDivergenceConfig(
            **self._loss_config))

    @parameterized.expand([(723,), (671,), (437,)])
    def test_loss_forward(self, length):
        # Unittest of loss forward
        enc_out = torch.rand(8, length, self._loss_config["num_classes"])
        enc_out_length = torch.tensor([20, 40, 101, length, 123, 143, 179, 201])
        labels = torch.randint(0, self._loss_config["num_classes"], (8, length))

        loss = self._kl_loss(enc_out, labels, enc_out_length)
        glog.info("Loss: {}".format(loss))


if __name__ == "__main__":
    unittest.main()
