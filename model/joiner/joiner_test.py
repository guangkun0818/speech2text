# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.05
""" Unittest of Joiner """

import glog
import unittest
import torch

from model.joiner.joiner import Joiner, JoinerConfig


class TestJoiner(unittest.TestCase):

    def setUp(self) -> None:
        self._unpruned_config = {
            "input_dim": 1024,
            "output_dim": 128,
            "activation": "relu",
            "prune_range": -1
        }
        self._unpruned_joiner = Joiner(config=JoinerConfig(
            **self._unpruned_config))

        self._pruned_config = {
            "input_dim": 1024,
            "output_dim": 128,
            "activation": "relu",
            "prune_range": 5
        }
        self._pruned_joiner = Joiner(config=JoinerConfig(**self._pruned_config))

    def test_unpruned_joiner_forward(self):
        # Unittest of forward of UnPruned Joiner.
        encoder_out = torch.rand(4, 200, 1024)
        encoder_out_lengths = torch.Tensor([197, 200, 65, 80])
        predictor_out = torch.rand(4, 16, 1024)
        tgts_length = torch.Tensor([8, 10, 9, 2])

        output, boundary, ranges = self._unpruned_joiner(
            encoder_out, encoder_out_lengths, predictor_out, tgts_length)

        # Output shape (B, T, U, D)
        self.assertEqual(output.shape[1], 200)
        self.assertEqual(output.shape[2], 16)
        self.assertEqual(output.shape[3], 128)
        glog.info("Output shape: {}".format(output.shape))

    def test_pruned_joiner_forward(self):
        # Unittest of forward of Pruned Joiner.
        encoder_out = torch.rand(4, 200, 1024)
        encoder_out_lengths = torch.Tensor([197, 200, 65, 80])
        predictor_out = torch.rand(4, 16, 1024)
        tgts_length = torch.Tensor([8, 10, 9, 2])
        tgts = torch.randint(1, 128, (4, 15))

        output, boundary, ranges = self._pruned_joiner(encoder_out,
                                                       encoder_out_lengths,
                                                       predictor_out,
                                                       tgts_length, tgts)

        # Output shape (B, T, U, D)
        self.assertEqual(output.shape[1], 200)
        self.assertEqual(output.shape[2], self._pruned_joiner.prune_range)
        self.assertEqual(output.shape[3], 128)
        glog.info("Output shape: {}".format(output.shape))

    def test_joiner_torchscript_export(self):
        # Unittest of joiner torchscript export
        self._pruned_joiner.train(False)
        ts_joiner = torch.jit.script(self._pruned_joiner)

        curr_hidden = torch.rand(1, 1, 1024)
        prev_token = torch.rand(1, 1, 1024)

        # (1, D)
        next_token_logits = ts_joiner.streaming_step(curr_hidden, prev_token)
        self.assertEqual(len(next_token_logits.shape), 2)
        self.assertEqual(next_token_logits.shape[0], 1)
        self.assertEqual(next_token_logits.shape[1], 128)


if __name__ == "__main__":
    unittest.main()
