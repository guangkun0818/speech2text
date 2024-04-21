# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.09
""" Unittest of Utilities """

import glog
import torch
import unittest

from parameterized import parameterized

from dataset.utils import TokenizerSetup
from model.predictor.predictor import Predictor
from model.joiner.joiner import Joiner, JoinerConfig
from model.utils import AsrMetric, AsrMetricConfig

# (B, T, D) = (2, 8, 5)
_LOGITS = torch.Tensor([[0.6, 0.2, 0.1, 0.1, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0,
                         0.0]]).unsqueeze(0).repeat(2, 1, 1)


class TestCtcMetric(unittest.TestCase):
    """ Unittest of Ctc task Metrics """

    def setUp(self) -> None:
        super(TestCtcMetric, self).__init__()
        # ground_truth_label = ("aabb", "abc")
        self._ground_truth = torch.Tensor([[1, 1, 2, 2], [1, 2, 3, 0]])
        # Set up char tokenizer
        self._char_config = {
            "type": "char",
            "config": {
                "labels": ["a", "b", "c"]
            }
        }
        self._char_tokenizer = TokenizerSetup(self._char_config)
        self._metric_config = {"decode_method": "ctc_greedy_search"}
        self._metrics = AsrMetric(self._char_tokenizer,
                                  config=AsrMetricConfig(**self._metric_config))

    # Params: (log probs, inputs_length, expect_wer)
    @parameterized.expand([(_LOGITS, torch.Tensor([8, 8]).long(), 0.5),
                           (_LOGITS, torch.Tensor([8, 3]).long(), 1.0)])
    def test_metric_call(self, log_probs, inputs_length, expect_wer):
        glog.info(log_probs.shape)
        wer = self._metrics(hidden_states=log_probs,
                            inputs_length=inputs_length,
                            ground_truth=self._ground_truth)
        self.assertEqual(wer, expect_wer)


class TestRnntMetric(unittest.TestCase):
    """ Unittest of Rnnt task metric """

    def setUp(self) -> None:
        self._tokenzier_config = {
            "type": "subword",
            "config": {
                "spm_model": "sample_data/spm/tokenizer.model",
                "spm_vocab": "sample_data/spm/tokenizer.vocab"
            }
        }
        self._tokenzier = TokenizerSetup(self._tokenzier_config)
        self._predictor_config = {
            "model": "Lstm",
            "config": {
                "num_symbols": 128,
                "output_dim": 512,
                "symbol_embedding_dim": 256,
                "num_lstm_layers": 3,
                "lstm_hidden_dim": 256,
            }
        }
        self._predictor = Predictor(config=self._predictor_config)

        # Input dim should be eq to predictor output_dim, output_dim
        # should be same as num_symbols of predictor.
        self._joiner_config = {
            "input_dim": 512,
            "output_dim": 128,
            "activation": "relu"
        }
        self._joiner = Joiner(config=JoinerConfig(**self._joiner_config))
        self._metric_config = {
            "metric": {
                "decode_method": "rnnt_greedy_search",
                "max_token_step": 1
            }
        }
        self._rnnt_metric = AsrMetric(
            self._tokenzier,
            config=AsrMetricConfig(**self._metric_config["metric"]),
            predictor=self._predictor,
            joiner=self._joiner)

    def test_rnnt_metric(self):
        # Unittest of call
        hidden_states = torch.rand(2, 8, 512)
        input_lengths = torch.Tensor([5, 8]).long()
        refers = torch.Tensor([[4, 4, 2, 2], [89, 2, 3, 0]])
        wer = self._rnnt_metric(hidden_states=hidden_states,
                                inputs_length=input_lengths,
                                ground_truth=refers)
        glog.info("Wer: {}".format(wer))


if __name__ == "__main__":
    unittest.main()
