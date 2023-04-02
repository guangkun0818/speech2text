# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.01
""" Unittest of Emformer """

import glog
import unittest
import torch
import torch.nn.functional as F

from model.encoder.emformer import Emformer, EmformerConfig


class TestEmformer(unittest.TestCase):

    def setUp(self) -> None:
        self._config = {
            "feats_dim": 80,
            "subsampling_rate": 4,
            "emformer_input_dim": 512,
            "num_heads": 8,
            "ffn_dim": 2048,
            "num_layers": 8,
            "segment_length": 4,
            "dropout": 0.1,
            "activation": "gelu",
            "left_context_length": 30,
            "right_context_length": 0,
            "max_memory_size": 0,
            "weight_init_scale_strategy": "depthwise",
            "tanh_on_mem": False,
            "output_dim": 1024
        }
        self._emformer = Emformer(config=EmformerConfig(**self._config))

    def test_emformer_forward(self):
        # Unittest of Emformer forward
        feats = torch.rand(3, 128, 80)
        lengths = torch.Tensor([64, 128, 80]).to(torch.int32)

        output, out_lengths = self._emformer(feats, lengths)
        glog.info("Output shape: {}".format(output.shape))
        glog.info("Out Lengths: {}".format(out_lengths))

    def test_enformer_stream_step(self):
        # Unittest of stream_step
        self._emformer.train(False)  # Eval stage
        feats = torch.rand(
            1, 60,
            80)  # Chunk size should comply with predefined segment_length
        states = []  # Initial states

        non_stream_output, _ = self._emformer(
            feats,
            torch.Tensor([60]).to(torch.int32))
        glog.info("Non-streaming output: {}".format(non_stream_output.shape))

        # Simu-streaming
        for i in range(3):
            output, states = self._emformer.streaming_step(
                feats[:, i * 20:20 * (i + 1), :], states)
            glog.info("Step {} Output shape: {}".format(i + 1, output.shape))

    def test_non_streaming_infer(self):
        # Unittest of non_streaming_ infer
        self._emformer.train(False)
        feats = torch.rand(1, 128, 80)
        dummy_length = torch.tensor([feats.shape[1]
                                    ]).to(feats.device).to(torch.int32)
        train_output, _ = self._emformer(feats, dummy_length)
        non_stream_infer = self._emformer.non_streaming_inference(feats)
        self.assertTrue(torch.allclose(train_output, non_stream_infer))

    def test_emformer_torchscript_export(self):
        # Unittest of Emformer torchscript export
        self._emformer.train(False)
        glog.info("Params of Emformer: {}".format(
            sum(p.numel() for p in self._emformer.parameters())))

        feats = torch.rand(3, 128, 80)
        lengths = torch.Tensor([64, 128, 80]).to(torch.int32)
        pt_output, _ = self._emformer(feats, lengths)

        ts_emformer = torch.jit.trace(self._emformer,
                                      example_inputs=[feats, lengths])
        ts_emformer = torch.jit.script(ts_emformer)
        ts_output, _ = ts_emformer(feats, lengths)

        # Precision check for forward
        self.assertTrue(torch.allclose(pt_output, ts_output))

        feats = torch.rand(1, 20, 80)
        state = ts_emformer.init_state()  # Initial states

        # Simu_Streaming & Precision check
        for _ in range(3):
            pt_stream_output, states = self._emformer.streaming_step(
                feats, state)
            ts_stream_output, states = ts_emformer.streaming_step(feats, state)

            # Precision check for streaming step
            self.assertTrue(torch.allclose(ts_stream_output, pt_stream_output))


if __name__ == "__main__":
    unittest.main()
