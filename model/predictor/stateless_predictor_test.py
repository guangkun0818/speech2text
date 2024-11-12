# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.07.28
""" Unittest of Stateless Predictor """

import glog
import unittest
import torch
import onnx

from model.predictor.stateless_predictor import StatelessPredictor
from model.predictor.stateless_predictor import StatelessPredictorConfig


class TestStatelessPredictor(unittest.TestCase):

    def setUp(self) -> None:
        self._config = {
            "num_symbols": 128,
            "output_dim": 256,
            "symbol_embedding_dim": 512,
            "context_size": 5
        }
        self._predictor = StatelessPredictor(config=StatelessPredictorConfig(
            **self._config))

    def test_predictor_forward(self):
        # Unittest of predictor forward
        inputs = torch.randint(1, 128, (5, 20))  #(B, U)
        lengths = torch.Tensor([10, 20, 8, 12, 11]).to(torch.int32)  # (B)

        states = self._predictor.init_state()
        # Training, only run for once
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
        batch_size = 4
        token_lens = 23
        self._predictor.train(False)
        inputs = torch.randint(1, 128, (batch_size, token_lens))  # (B, U)
        lengths = torch.Tensor([token_lens] * batch_size).to(torch.int32)  # (B)

        ts_predictor = torch.jit.script(self._predictor)
        states = ts_predictor.init_state()
        output, lengths, states = ts_predictor(inputs, lengths, states)
        # Simu-streaming
        s_output = []
        states = self._predictor.init_state(batch_size)
        start_token = torch.zeros(batch_size, 1).to(torch.int32)
        step_out, states = ts_predictor.streaming_step(start_token, states)

        for i in range(token_lens):
            # Unittest of streaming step
            step_out, states = ts_predictor.streaming_step(
                inputs[:, i:i + 1], states)
            s_output.append(step_out)

        # Presicion check
        s_output = torch.concat(s_output, dim=1)
        self.assertTrue(
            torch.allclose(output[:, 1:], s_output, rtol=3e-5, atol=1e-6))

    def test_predictor_sherpa_onnx_export(self):
        # Unittest of Onnx export
        export_filename = "test_logs"
        self._predictor._sherpa_onnx_export(export_filename)

    def test_predictor_sherpa_onnx_export(self):
        # Unittest of Onnx export
        export_filename = "test_logs"
        self._predictor._mnn_onnx_export(export_filename)


if __name__ == "__main__":
    unittest.main()
