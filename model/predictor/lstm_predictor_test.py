# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.05
""" Unittest of LSTM based Predictor """

import glog
import unittest
import torch

from model.predictor.lstm_predictor import LstmPredictor
from model.predictor.lstm_predictor import LstmPredictorConfig


class TestLstmPredictor(unittest.TestCase):

    def setUp(self) -> None:
        self._config = {
            "num_symbols": 128,
            "output_dim": 1024,
            "symbol_embedding_dim": 512,
            "num_lstm_layers": 3,
            "lstm_hidden_dim": 512,
            "lstm_layer_norm": True,
            "lstm_layer_norm_epsilon": 1e-3,
            "lstm_dropout": 0.3
        }
        self._predictor = LstmPredictor(config=LstmPredictorConfig(
            **self._config))

    def test_predictor_forward(self):
        # Unittest of predictor forward
        inputs = torch.randint(1, 128, (5, 20))  # (B, U)
        lengths = torch.Tensor([10, 20, 8, 12, 11]).to(torch.int32)  # (B)
        states = self._predictor.init_state()
        for i in range(2):
            output, lengths, states = self._predictor(inputs, lengths, states)

        glog.info("Input shape: {}".format(inputs.shape))
        glog.info("Output shape: {}".format(output.shape))
        glog.info("Lengths shape: {}".format(lengths))

    def test_left_padding(self):
        # Unittest of left_padding
        inputs = torch.randint(1, 120, (5, 20))  # (B, U)
        inputs = self._predictor._left_padding(inputs)
        self.assertEqual(self._config["num_symbols"] - 1,
                         self._predictor.sos_token)
        self.assertEqual(self._predictor.blank_token, 0)
        self.assertTrue(
            torch.allclose(
                inputs[:, :1],
                torch.tensor([[0] for i in range(5)]).to(torch.int32)))

    def test_predictor_torchscript_export(self):
        # Unittest of Predictor export
        self._predictor.train(False)
        inputs = torch.randint(1, 128, (5, 20))  # (B, U)
        lengths = torch.Tensor([10, 20, 8, 12, 11]).to(torch.int32)  # (B)

        ts_predictor = torch.jit.script(self._predictor)
        states = ts_predictor.init_state()
        for i in range(2):
            output, lengths, states = ts_predictor(inputs, lengths, states)
            # Unittest of streaming step
            output, states = ts_predictor.streaming_step(
                inputs[i:i + 1, i:i + 1], states)


if __name__ == "__main__":
    unittest.main()
