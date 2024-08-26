# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.26
""" Unittest of Zipformer impl. """

import glog
import torch
import onnx
import unittest

from parameterized import parameterized
from model.encoder.zipformer import Zipformer2, Zipformer2Config


class TestZipformer(unittest.TestCase):

    def setUp(self) -> None:
        self._config = {
            "feature_dim": 80,
            "downsampling_factor": [1, 2, 4, 8, 4, 2],
            "num_encoder_layers": [2, 2, 2, 2, 2, 2],
            "feedforward_dim": [512, 768, 768, 768, 768, 768],
            "encoder_dim": [192, 256, 256, 256, 256, 256],
            "encoder_unmasked_dim": [192, 192, 192, 192, 192, 192],
            "num_heads": [4, 4, 4, 8, 4, 4],
            "query_head_dim": 32,
            "value_head_dim": 12,
            "pos_head_dim": 4,
            "pos_dim": 48,
            "cnn_module_kernel": [31, 31, 15, 15, 15, 31],
            "causal": True,
            "chunk_size": [16, 32, 64, -1],
            "left_context_frames": [64, 128, 256, -1],
            "for_ctc": True,
            "num_tokens": 1000
        }
        self._zipformer = Zipformer2(config=Zipformer2Config(**self._config))

    def test_zipformer_infos(self):
        glog.info("Params of Zipformer: {}".format(
            sum(p.numel() for p in self._zipformer.parameters())))

    @parameterized.expand([(10,), (20,)])
    def test_zipformer_forward(self, batch_size):
        # zipformer forward
        lengths = torch.randint(1, 400, (batch_size,))
        feats = torch.rand(batch_size, int(lengths.max()), 80)
        outputs, outputs_length = self._zipformer(feats, lengths)
        glog.info(outputs.shape)
        glog.info(outputs_length)

    @parameterized.expand([(10,), (20,)])
    def test_zipformer_streaming_forward(self, batch_size):
        # zipformer forward
        lengths = torch.randint(1, 400, (batch_size,))
        feats = torch.rand(batch_size, int(lengths.max()), 80)
        outputs, outputs_length = self._zipformer.streaming_forward(
            feats, lengths, chunk_size=[32], left_context_frames=[128])
        glog.info(outputs.shape)
        glog.info(outputs_length)

    def test_zipformer_streaming_step(self):
        # zipformer forward
        self._zipformer.causal = True
        self._zipformer.chunk_size = [32]
        self._zipformer.left_context_frames = [128]

        chunk_feats = torch.rand(1, 77, 80)
        states = self._zipformer.get_init_states()
        outputs, new_states = self._zipformer.streaming_step(
            chunk_feats, states)
        glog.info(outputs.shape)


if __name__ == "__main__":
    unittest.main()
