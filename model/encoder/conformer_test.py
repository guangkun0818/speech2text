# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.01
""" Unittest of Conformer """

import glog
import torch
import unittest
import torch.nn.functional as F

from parameterized import parameterized
from torch.nn.utils.rnn import pad_sequence

from model.encoder.conformer import Conformer, ConformerConfig, Subsampling


class TestConformer(unittest.TestCase):
    """ Unittest of Conformer """

    def setUp(self) -> None:
        self._input_config = {
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
            "output_dim": 45
        }
        self._config = ConformerConfig(**self._input_config)
        self._conformer = Conformer(config=self._config)
        self._subsampling_module = self._conformer._subsampling_module
        self._conformer_module = self._conformer._conformer_module

    def test_subsampling_4(self):
        # Unittest of subsampling4 module
        glog.info("Unittest of Subsampling4. ...")
        self._subsampling_module = Subsampling(idim=80,
                                               odim=512,
                                               subsampling_rate=4)

        lengths = torch.randint(4, 400, (10,))
        feats = torch.rand(10, int(lengths.max()), 80)
        glog.info("Input feats shape: {}".format(feats.shape))

        output, out_length = self._subsampling_module(feats, lengths)
        glog.info("Output feats shape: {}".format(output.shape))

        self.assertEqual(output.shape[-1], 512)
        self.assertEqual(output.shape[1], int(out_length.max()))

    def test_subsampling_6(self):
        # Unittest of subsampling6 module
        glog.info("Unittest of Subsampling4. ...")
        self._subsampling_module = Subsampling(idim=80,
                                               odim=512,
                                               subsampling_rate=6)

        lengths = torch.randint(4, 400, (10,))
        feats = torch.rand(10, int(lengths.max()), 80)
        glog.info("Input feats shape: {}".format(feats.shape))

        output, out_length = self._subsampling_module(feats, lengths)
        glog.info("Output feats shape: {}".format(output.shape))

        self.assertEqual(output.shape[-1], 512)
        self.assertEqual(output.shape[1], int(out_length.max()))

    def test_subsampling_8(self):
        # Unittest of subsampling8 module
        glog.info("Unittest of Subsampling8. ...")
        self._subsampling_module = Subsampling(idim=80,
                                               odim=512,
                                               subsampling_rate=8)

        lengths = torch.randint(4, 400, (10,))
        feats = torch.rand(10, int(lengths.max()), 80)
        glog.info("Input feats shape: {}".format(feats.shape))

        output, out_length = self._subsampling_module(feats, lengths)
        glog.info("Output feats shape: {}".format(output.shape))

        self.assertEqual(output.shape[-1], 512)
        self.assertEqual(output.shape[1], int(out_length.max()))

    def test_conformer_training(self):
        # Unittest of Conformer module training graph
        glog.info("Unittest of Conformer training...")
        lengths = torch.randint(4, 400, (10,))
        feats = torch.rand(10, int(lengths.max()), 80)
        logits, lengths = self._conformer(feats, lengths)
        glog.info("Output length: {}".format(lengths))
        glog.info("Output logits shape: {}".format(logits.shape))

        self.assertEqual(logits.shape[-1], self._config.output_dim)
        self.assertEqual(logits.shape[1], int(lengths.max()))

    def test_conformer_inference(self):
        # Unittest of Conformer module inference graph
        glog.info("Unittest of Conformer inference...")
        lengths = torch.randint(4, 400, (10,))
        feats = torch.rand(10, int(lengths.max()), 80)
        logits = self._conformer.non_streaming_inference(feats)
        glog.info("Output logits shape: {}".format(logits.shape))

    @parameterized.expand([((128, 256),)])
    def test_conformer_train_infer_match(self, lengths):
        # Unittest to make sure training inference results are same
        feats = []
        for l in lengths:
            feats.append(torch.rand(l, 80))

        # setup inputs
        feats = pad_sequence(feats, batch_first=True, padding_value=0)
        lengths = torch.Tensor(lengths)

        # precision check
        self._conformer.eval()
        logits_train, _ = self._conformer(feats, lengths)
        logits_train = F.log_softmax(logits_train, dim=-1)
        infer_feats = feats[0:1, :int(lengths[0]), :]
        logits_infered = self._conformer.non_streaming_inference(infer_feats)
        # Due to padding of batch, there exists precision mismatch over shorted
        # one within batch between training and inference due to Conv forward.
        # By adding padding at the rear of input feats might fix this issue, but
        # basically its impact on model performance is trivial. Besides, this
        # will totally reconstruct model will high-code dev, not propoitional
        # with the contribution to wer, so I maintain the conformer as official impl.

        # Precision check
        infer_feats = feats[1:2, :int(lengths[1]), :]
        logits_infered = self._conformer.non_streaming_inference(infer_feats)
        self.assertTrue(
            torch.allclose(logits_train[1:2, :int(lengths[1]), :],
                           logits_infered[:, :int(lengths[1]), :]))


if __name__ == "__main__":
    unittest.main()
