# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.03.28
""" Unittest of Frontend of """

import glog
import unittest
import torchaudio
import torch

from dataset.frontend.frontend import FeatType


class KaldiFbankTest(unittest.TestCase):
    """ Unittest of KaldiWaveFeature frontend """

    def setUp(self) -> None:
        self._test_data_1 = "sample_data/data/wavs/251-136532-0007.wav"
        self._test_data_2 = "sample_data/data/wavs/1462-170138-0015.wav"

        self._config = {
            "feat_type": "fbank",
            "feat_config": {
                "num_mel_bins": 80,
                "frame_length": 25,
                "frame_shift": 10,
                "dither": 0.0,
                "samplerate": 16000,
            },
        }
        self._kaldi_frontend = FeatType[self._config["feat_type"]].value(
            **self._config["feat_config"])

    def test_frontend_wave_feature(self):
        # Frontend forward unittest
        pcms, _ = torchaudio.load(self._test_data_1)
        feats = self._kaldi_frontend(pcms)
        glog.info("Fbank feature: {}".format(feats.shape))
        self.assertEqual(feats.shape[-1], self._kaldi_frontend._num_mel_bins)

    def test_frontend_torchscript(self):
        # Frontend torchscript export unittest

        pcms = torch.rand(1, 32000)
        torchscript_frontend = torch.jit.trace(self._kaldi_frontend,
                                               example_inputs=pcms)
        torchscript_frontend = torch.jit.script(torchscript_frontend)

        # Torchscript frontend precision check
        pcms = torch.rand(1, 48000)
        pt_feats = self._kaldi_frontend(pcms)
        ts_feats = torchscript_frontend(pcms)
        glog.info("Torchscript output :{}".format(ts_feats.shape))
        glog.info("Checkpoint output: {}".format(pt_feats.shape))
        self.assertTrue(torch.allclose(pt_feats, ts_feats))


class TestLhotseKaldiFeatFbank(unittest.TestCase):
    """ Unittest of LhotseKaldiFeatFbank and FeatType """

    def setUp(self) -> None:
        self._test_data_1 = "sample_data/data/wavs/251-136532-0007.wav"
        self._test_data_2 = "sample_data/data/wavs/1462-170138-0015.wav"

        self._config = {
            "feat_type": "lhotes_fbank",
            "feat_config": {
                "num_mel_bins": 80,
            },
        }

        self._frontend = FeatType[self._config["feat_type"]].value(
            **self._config["feat_config"])

    def test_frontend_wave_feature(self):
        # Frontend forward unittest
        pcms, _ = torchaudio.load(self._test_data_1,
                                  normalize=self._frontend.pcm_normalize)
        feats = self._frontend(pcms)
        glog.info("Fbank feature: {}".format(feats.shape))
        self.assertEqual(feats.shape[-1], self._frontend.feat_dim)


class TestTorchScriptKaldiWaveFeature(unittest.TestCase):

    def setUp(self) -> None:
        self._test_data_1 = "sample_data/data/wavs/251-136532-0007.wav"
        self._test_data_2 = "sample_data/data/wavs/1462-170138-0015.wav"

        self._config = {
            "feat_type": "torchscript_fbank",
            "feat_config": {
                "torchscript": "sample_data/model/frontend.script",
                "num_mel_bins": 64,
            },
        }

        self._frontend = FeatType[self._config["feat_type"]].value(
            **self._config["feat_config"])

    def test_frontend_wave_feature(self):
        # Frontend forward unittest
        pcms, _ = torchaudio.load(self._test_data_1,
                                  normalize=self._frontend.pcm_normalize)
        feats = self._frontend(pcms)
        glog.info("Fbank feature: {}".format(feats.shape))
        self.assertEqual(feats.shape[-1], self._frontend.feat_dim)


if __name__ == "__main__":
    unittest.main()
