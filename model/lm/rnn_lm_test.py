# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.27
""" Unittest of Rnn lm """

import glog
import unittest
import torch

from model.lm.rnn_lm import RnnLm, RnnLmConfig


class UnittestRnnLm(unittest.TestCase):

    def setUp(self) -> None:
        self._config = {
            "num_symbols": 128,
            "symbol_embedding_dim": 512,
            "num_rnn_layer": 3,
            "dropout": 0.0,
            "bidirectional": False,
        }

        self._rnn_lm = RnnLm(config=RnnLmConfig(**self._config))

    def test_zipformer_infos(self):
        glog.info("Params of RnnLm: {}".format(
            sum(p.numel() for p in self._rnn_lm.parameters())))

    def test_rnn_lm_forward(self):
        vocab_size = self._config["num_symbols"]
        batch_size = 256

        x_lens = torch.randint(1, 400, (batch_size,))
        x = torch.randint(1, vocab_size, (256, int(x_lens.max())))

        outputs, outputs_length = self._rnn_lm(x, x_lens)
        glog.info(outputs.shape)
        glog.info(outputs_length)

    def test_rnn_lm_score(self):
        vocab_size = self._config["num_symbols"]
        batch_size = 256

        x_lens = torch.randint(1, 400, (batch_size,))
        x = torch.randint(1, vocab_size, (256, int(x_lens.max())))

        scores = self._rnn_lm.score(x, x_lens)
        glog.info(scores)

    def test_rnn_lm_score_step(self):
        vocab_size = self._config["num_symbols"]
        beam_size = 4
        states = self._rnn_lm.init_states(beam_size)

        tokens = torch.randint(1, vocab_size, (beam_size,))

        log_probs, states = self._rnn_lm.score_step(tokens, states)
        glog.info(log_probs.shape)
        self.assertEqual(log_probs.shape[1], vocab_size)


if __name__ == "__main__":
    unittest.main()
