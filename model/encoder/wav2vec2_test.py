# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.02
""" Unittest of Wav2Vec2 """

import glog
import unittest
import torch
import torch.nn.functional as F

from model.encoder.wav2vec2 import Wav2Vec2CustomizedConfig, Wav2Vec2Encoder


class Wav2Vec2EncoderTest(unittest.TestCase):
    """ Unittest of Wav2Vec2 Encoder """

    def setUp(self) -> None:
        self._config = {
            "pretrained_model": "facebook/wav2vec2-base-960h",
            "hidden_size": 768,
            "label_dim": 45
        }
        self._wav2vec2 = Wav2Vec2Encoder(config=Wav2Vec2CustomizedConfig(
            **self._config))

    def test_compute_logits_length(self):
        # Unittest of compute_logits_length
        lengths = torch.LongTensor([16000, 32000, 16080, 32160, 16079, 32158])
        glog.info("Pcms lengths: {}".format(lengths))

        output_lengths = self._wav2vec2._compute_logits_length(lengths=lengths)
        glog.info("Output lengths: {}".format(output_lengths))

        self.assertTrue(
            torch.allclose(output_lengths,
                           torch.LongTensor([49, 99, 50, 100, 49, 100])))

    def test_zero_mean_unit_var_norm(self):
        # Unittest of data normalization
        pcms = torch.rand(2, 32000)
        lengths = torch.LongTensor([27000, 32000])
        pcms = self._wav2vec2._zero_mean_unit_var_norm(pcms, lengths)

        pcms = torch.rand(1, 16000)
        lengths = torch.LongTensor([16000])
        pcms = self._wav2vec2._zero_mean_unit_var_norm(pcms, lengths)

    def test_forward_wav2vec2(self):
        # Unittest of Wav2Vec2 training graph
        pcms = torch.rand(2, 32000)
        pcms_lengths = torch.LongTensor([32000, 16000])
        glog.info("Pcms shape: {}".format(pcms.shape))
        glog.info("Pcms lengths: {}".format(pcms_lengths))

        output, lengths = self._wav2vec2(pcms, pcms_lengths)
        glog.info("Output shape: {}".format(output.shape))
        glog.info("Output lengths: {}".format(lengths))

        self.assertEqual(output.shape[1], max(lengths).item())

    def test_inference_wav2vec2(self):
        pcms = torch.rand(1, 32000)
        pcms_lengths = torch.LongTensor([32000])
        glog.info("Pcms shape: {}".format(pcms.shape))
        glog.info("Pcms lengths: {}".format(pcms_lengths))

        train_output, _ = self._wav2vec2(pcms, pcms_lengths)
        train_output = F.log_softmax(train_output, dim=-1)
        infer_output = self._wav2vec2.non_streaming_inference(pcms)
        self.assertTrue(torch.allclose(train_output, infer_output))


if __name__ == "__main__":
    unittest.main()
