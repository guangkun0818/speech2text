# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.11
""" Unittest of Continuous Integrate-Fire Layer """

import glog
import unittest
import torch

from parameterized import parameterized
from model.cif.cif_layer import CifLayer, CifLayerConfig
from model.encoder.conformer import Conformer, ConformerConfig


class TestCifLayer(unittest.TestCase):

    def setUp(self) -> None:
        self._cif_config = {
            "idim": 512,
            "l_pad": 0,
            "r_pad": 0,
            "dropout": 0.0,
            "threshold": 1.0,
            "smooth_factor": 1.0,
            "noise_threshold": 0.0,
            "tail_threshold": 0.45
        }

        self._cif_layer = CifLayer(config=CifLayerConfig(**self._cif_config))

        self._conformer_config = {
            "bn_cmvn": False,
            "feats_dim": 80,
            "subsampling_rate": 4,
            "input_dim": 512,
            "num_heads": 8,
            "ffn_dim": 2048,
            "num_layers": 8,
            "depthwise_conv_kernel_size": 31,
            "dropout": 0.0,
            "use_group_norm": False,
            "convolution_first": False,
            "output_dim": 512
        }
        self._conformer = Conformer(config=ConformerConfig(
            **self._conformer_config))

    @parameterized.expand([(10, 100), (23, 195), (32, 399)])
    def test_cif_layer_forward(self, batchsize, timestep):
        # Unittest of CifLayer Conformer subsampling4
        lengths = torch.randint(7, timestep, (batchsize,))
        feats = torch.rand(batchsize, int(lengths.max()), 80)
        tgt_len = torch.randint(0, timestep, (batchsize,))
        tgt = torch.randint(0, 48, (batchsize, int(tgt_len.max()))).long()

        glog.info("Lengths shape: {}".format(lengths.shape))
        glog.info("Conformer input feats shape: {}".format(feats.shape))
        encoder_out, encoder_out_length = self._conformer(feats, lengths)
        glog.info("Conformer output feats shape: {}".format(encoder_out.shape))
        glog.info("Conformer output length shape: {}".format(
            encoder_out_length.shape))
        glog.info("Conformer output length: {}".format(encoder_out_length))

        acoustic_embeds, cif_peak, token_num_hat, alphas = self._cif_layer(
            encoder_out, encoder_out_length, tgt, tgt_len)
        glog.info("acoustic_embeds shape: {}".format(acoustic_embeds.shape))
        glog.info("cif_peak shape: {}".format(cif_peak.shape))
        glog.info("token_num_hat shape: {}".format(token_num_hat.shape))
        glog.info("token_num_hat: {}".format(token_num_hat))
        glog.info("alphas shape: {}".format(alphas.shape))


if __name__ == "__main__":
    unittest.main()
