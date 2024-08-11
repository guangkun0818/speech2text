# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.11
""" Unittest of BestRQLayer """

import unittest
import glog
import torch
import torchaudio

from parameterized import parameterized
from dataset.frontend.frontend import KaldiWaveFeature
from model.ssl.best_rq import BestRQLayer, BestRQLayerConfig, MaskingStrategyConfig
from model.encoder.conformer import Subsampling


class TestBestRQLayer(unittest.TestCase):
    """ Unittest of Best-RQ Layer """

    def setUp(self) -> None:
        self._bestrq_layer_config = {
            "pre_post_norm": False,
            "cnn_kernel_size": [3, 3],
            "cnn_stride": [2, 2],
            "feat_dim": 80,
            "num_codebooks": 16,
            "codebook_dim": 16,
            "codebook_size": 8192,
            "label_basis": "euclidean"
        }
        self._masking_config = {
            "mask_proportion": 0.45,
            "mean_span_length": 10,
            "span_select_type": "static",
            "span_length_float_rate": None,
            "min_num_spans": 1,
            "no_overlap": True,
            "min_space": 1,
            "seed": None
        }
        self._bestrq = BestRQLayer(
            layer_config=BestRQLayerConfig(**self._bestrq_layer_config),
            masking_config=MaskingStrategyConfig(**self._masking_config))
        self._frontend = KaldiWaveFeature(num_mel_bins=80)

    def test_bestrq_layer_forward(self):
        # Unittest of BestRQLayer forward
        test_data = "sample_data/data/wavs/1462-170138-0015.wav"
        test_pcm, _ = torchaudio.load(test_data, normalize=True)
        test_feats = self._frontend(test_pcm).unsqueeze(0)
        test_lengths = torch.tensor([test_feats.shape[1]]).long()

        batch = self._bestrq(test_feats, test_feats, test_lengths)
        glog.info("Masked feats shape: {}".format(batch["masked_feats"].shape))
        glog.info("Labels shape: {}".format(
            batch["labels"].shape))  # (num codebooks, B, T // 4)
        glog.info("Masked proportion: {}".format(batch["masked_dim"].sum() /
                                                 batch["masked_dim"].numel()))
        self.assertEqual(batch["labels"].shape[1:], batch["masked_dim"].shape)

    def test_get_subsampling_arrangment(self):
        # Unittest of BestRQLayer get subsampling arrangment
        lengths = torch.randint(7, 400, (10,))
        feats = torch.rand(10, int(lengths.max()), 80)

        sub_frame_arr, label_lengths, labels = self._bestrq._get_subsampling_arrangment(
            feats, lengths)
        glog.info("Input feats shape: {}".format(feats.shape))
        glog.info("Sub_frame_arr shape: {}".format(sub_frame_arr.shape))
        glog.info("Subsampled length: {}".format(label_lengths))
        glog.info("Label shape: {}".format(
            labels.shape))  # (num_codebooks, B, T // 4)

        self.assertEqual(sub_frame_arr.shape[1], labels.shape[2])

    def test_random_mask(self):
        # Unittest of BestRQLayer random mask
        lengths = torch.randint(7, 400, (10,))
        feats = torch.rand(10, int(lengths.max()), 80)
        glog.info("Randint lengths: {}".format(lengths))

        sub_frame_arr, subsampled_len, labels = self._bestrq._get_subsampling_arrangment(
            feats, lengths)
        masked_feats, masked_dim = self._bestrq._random_mask(
            feats, sub_frame_arr, subsampled_len)
        glog.info("Masked feats shape: {}".format(masked_feats.shape))
        glog.info("Masked dimension shape: {}".format(masked_dim.shape))

    @parameterized.expand([(23, 199), (32, 399)])
    def test_conformer_subsampling_4(self, batchsize, timestep):
        # Unittest of BestRQLayer conformer subsampling4
        self._subsampling_module = Subsampling(idim=80,
                                               odim=512,
                                               subsampling_rate=4)
        lengths = torch.randint(7, timestep, (batchsize,))
        feats = torch.rand(batchsize, int(lengths.max()), 80)

        batch = self._bestrq(feats, feats, lengths)
        glog.info("Best_rq masked feats shape: {}".format(
            batch["masked_feats"].shape))
        glog.info("Best_rq labels shape: {}".format(batch["labels"].shape))
        glog.info("Best_rq masked_dim shape: {}".format(
            batch["masked_dim"].shape))

        feats = batch["masked_feats"]
        glog.info("Conformer input feats shape: {}".format(feats.shape))
        output, out_length = self._subsampling_module(feats, lengths)
        glog.info("Conformer output feats shape: {}".format(output.shape))
        glog.info("masked proportion: {}".format(batch["masked_dim"].sum() /
                                                 out_length.sum()))

        self.assertEqual(output.shape[-1], 512)
        self.assertEqual(output.shape[1], int(out_length.max()))
        self.assertEqual(batch["labels"].shape[1:], output.shape[:2])

    def test_compute_mask_indices(self):
        sub_timestep = 100
        padding_num = torch.randint(7, 100, (1,)).item()
        glog.info("Padding_num: {}".format(padding_num))
        mask = self._bestrq._compute_mask_indices(sub_timestep, padding_num)
        glog.info("Mask: {}".format(mask))
        glog.info("Mask length: {}".format(len(mask)))


if __name__ == "__main__":
    unittest.main()
