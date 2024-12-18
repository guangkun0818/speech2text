# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.01
""" Unittest of Dataset """

import glog
import json
import unittest

from parameterized import parameterized
from torch.utils.data import DataLoader
from dataset.dataset import AsrTrainDataset, AsrEvalDataset, AsrTestDataset
from dataset.dataset import SslTrainDataset, SslEvalDataset
from dataset.dataset import LmDataset
from dataset.dataset import (asr_collate_fn, asr_test_collate_fn,
                             ssl_collate_fn, lm_collate_fn)
from dataset.utils import TokenizerSetup


class TestAsrFbankSubwordDataset(unittest.TestCase):
    """ Unittest of Asr train, eval Dataset, fbank and subword specified. 
    """

    def setUp(self):
        self._tokenizer_config = {
            "type": "subword",
            "config": {
                "spm_model": "sample_data/spm/tokenizer.model",
                "spm_vocab": "sample_data/spm/tokenizer.vocab"
            }
        }

        self._dataset_config = {
            "train_data": "sample_data/asr_train_data.json",
            "eval_data": "sample_data/asr_eval_data.json",
            "noise_data": "sample_data/noise_data.json",
            "apply_segment": False,
            "dur_min_filter": 0.0,
            "dur_max_filter": 20.0,
            "batch_size": 256,
            "feat_type": "fbank",
            "feat_config": {
                "num_mel_bins": 80,
                "frame_length": 25,
                "frame_shift": 10,
                "dither": 0.0,
                "samplerate": 16000
            },
            "data_aug_config": {
                "use_speed_perturb": True,
                "use_spec_aug": True,
                "use_add_noise": True,
                "add_noise_proportion": 0.5,
                "add_noise_config": {
                    "min_snr_db": 10,
                    "max_snr_db": 50,
                    "max_gain_db": 300.0,
                },
                "use_mix_feats": True,
                "mix_feats_proportion": 0.5,
                "mix_feats_config": {
                    "snrs": [10, 20]
                }
            }
        }

        self._tokenzier = TokenizerSetup(self._tokenizer_config)
        self._train_dataset = AsrTrainDataset(self._dataset_config,
                                              self._tokenzier)
        self._eval_dataset = AsrEvalDataset(self._dataset_config,
                                            self._tokenzier)

    def test_dataset_info(self):
        glog.info("Total duration: {}".format(
            self._train_dataset.total_data_amount))
        glog.info("Min duration: {}".format(self._train_dataset.lower_bound))
        glog.info("Max duration: {}".format(self._train_dataset.high_bound))

    # Batch_size from 40/101/341
    @parameterized.expand([(40,), (101,), (341,)])
    def test_train_dataset(self, batch_size):
        glog.info("Unittest of train_dataset, fbank and subword specified.")
        count = 0
        dataloader = DataLoader(dataset=self._train_dataset,
                                batch_size=batch_size,
                                collate_fn=asr_collate_fn)
        for i, batch in enumerate(dataloader):
            count += 1
            glog.info("feat: {}".format(batch["feat"].shape))
            glog.info("feat_length: {}".format(batch["feat_length"]))
            glog.info("label: {}".format(batch["label"].shape))
            glog.info("label_length: {}".format(batch["label_length"]))
            glog.info(count)

    # Batch_size from 40/101/341
    @parameterized.expand([(40,), (101,), (341,)])
    def test_eval_dataset(self, batch_size):
        glog.info("Unittest of eval_dataset, fbank and subword specified.")
        count = 0
        dataloader = DataLoader(dataset=self._eval_dataset,
                                batch_size=batch_size,
                                collate_fn=asr_collate_fn)
        for i, batch in enumerate(dataloader):
            count += 1
            glog.info("feat: {}".format(batch["feat"].shape))
            glog.info("feat_length: {}".format(batch["feat_length"]))
            glog.info("label: {}".format(batch["label"].shape))
            glog.info("label_length: {}".format(batch["label_length"]))
            glog.info(count)


class TestAsrPcmCharDataset(unittest.TestCase):
    """ Unittest of Asr train, eval Dataset, Pcm and char specified. 
    """

    def setUp(self):
        self._tokenizer_config = {
            "type": "char",
            "config": {
                "labels": [
                    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
                    "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x",
                    "y", "z", "'", " "
                ]
            }
        }

        self._dataset_config = {
            "train_data": "sample_data/asr_train_data.json",
            "eval_data": "sample_data/asr_eval_data.json",
            "noise_data": "sample_data/noise_data.json",
            "apply_segment": False,
            "dur_min_filter": 0.0,
            "dur_max_filter": 20.0,
            "batch_size": 256,
            "feat_type": "pcm",
            "feat_config": {
                "dummy": -1
            },
            "data_aug_config": {
                "use_speed_perturb": True,
                "use_spec_aug": False,
                "use_add_noise": True,
                "add_noise_proportion": 0.5,
                "add_noise_config": {
                    "min_snr_db": 10,
                    "max_snr_db": 50,
                    "max_gain_db": 300.0,
                },
                "use_mix_feats": False,
                "mix_feats_proportion": 0.5,
                "mix_feats_config": {
                    "snrs": [10, 20]
                }
            }
        }

        self._tokenzier = TokenizerSetup(self._tokenizer_config)
        self._train_dataset = AsrTrainDataset(self._dataset_config,
                                              self._tokenzier)
        self._eval_dataset = AsrEvalDataset(self._dataset_config,
                                            self._tokenzier)

    def test_dataset_info(self):
        glog.info("Total duration: {}".format(
            self._train_dataset.total_data_amount))
        glog.info("Min duration: {}".format(self._train_dataset.lower_bound))
        glog.info("Max duration: {}".format(self._train_dataset.high_bound))

    # Batch_size from 40/101/341
    @parameterized.expand([(40,), (101,), (341,)])
    def test_train_dataset(self, batch_size):
        glog.info("Unittest of train_dataset, pcm and char specified.")
        count = 0
        dataloader = DataLoader(dataset=self._train_dataset,
                                batch_size=batch_size,
                                collate_fn=asr_collate_fn)
        for i, batch in enumerate(dataloader):
            count += 1
            glog.info("feat: {}".format(batch["feat"].shape))
            glog.info("feat_length: {}".format(batch["feat_length"]))
            glog.info("label: {}".format(batch["label"].shape))
            glog.info("label_length: {}".format(batch["label_length"]))
            glog.info(count)

    # Batch_size from 40/101/341
    @parameterized.expand([(40,), (101,), (341,)])
    def test_eval_dataset(self, batch_size):
        glog.info("Unittest of eval_dataset, pcm and char specified.")
        count = 0
        dataloader = DataLoader(dataset=self._eval_dataset,
                                batch_size=batch_size,
                                collate_fn=asr_collate_fn)
        for i, batch in enumerate(dataloader):
            count += 1
            glog.info("feat: {}".format(batch["feat"].shape))
            glog.info("feat_length: {}".format(batch["feat_length"]))
            glog.info("label: {}".format(batch["label"].shape))
            glog.info("label_length: {}".format(batch["label_length"]))
            glog.info(count)


class TestAsrTestDataset(unittest.TestCase):
    """ Unittest of Asr test Dataset. """

    def setUp(self):

        self._config = {
            "test_data": "sample_data/asr_train_data.json",
            "config": {
                "batch_size": 24,
                "apply_segment": False,
                "feat_type": "torchscript_fbank",
                "feat_config": {
                    "torchscript": "sample_data/model/frontend.script",
                    "num_mel_bins": 64,
                },
            }
        }

        self._test_dataset = AsrTestDataset(
            testset_config=self._config["config"],
            testset_json=self._config["test_data"])

    # Batch_size == 1
    def test_asrtest_dataset(self):
        glog.info("Unittest of test_dataset.")
        count = 0
        dataloader = DataLoader(dataset=self._test_dataset,
                                batch_size=self._config["config"]["batch_size"],
                                collate_fn=asr_test_collate_fn)
        for i, batch in enumerate(dataloader):
            count += 1
            glog.info("audio_filepath: {}".format(batch["audio_filepath"]))
            glog.info("feat: {}".format(batch["feat"].shape))
            glog.info("feat_length: {}".format(batch["feat_length"]))
            glog.info("text: {}".format(batch["text"]))
            glog.info(count)


class TestSslTrainEvalDataset(unittest.TestCase):
    """ Unittest of SSL train eval dataset. """

    def setUp(self) -> None:
        self._dataset_config = {
            "train_data": "sample_data/asr_train_data.json",
            "eval_data": "sample_data/asr_eval_data.json",
            "noise_data": "sample_data/noise_data.json",
            "apply_segment": False,
            "dur_min_filter": 0.0,
            "dur_max_filter": 20.0,
            "batch_size": 256,
            "feat_type": "fbank",
            "feat_config": {
                "num_mel_bins": 80,
                "frame_length": 25,
                "frame_shift": 10,
                "dither": 0.0,
                "samplerate": 16000
            },
            "data_aug_config": {
                "use_speed_perturb": True,
                "use_spec_aug": True,
                "use_add_noise": True,
                "add_noise_proportion": 0.5,
                "add_noise_config": {
                    "min_snr_db": 10,
                    "max_snr_db": 50,
                    "max_gain_db": 300.0,
                },
                "use_mix_feats": True,
                "mix_feats_proportion": 0.5,
                "mix_feats_config": {
                    "snrs": [10, 20]
                }
            }
        }
        self._train_dataset = SslTrainDataset(self._dataset_config)
        self._eval_dataset = SslEvalDataset(self._dataset_config)

    def test_dataset_info(self):
        glog.info("Total duration: {}".format(
            self._train_dataset.total_data_amount))
        glog.info("Min duration: {}".format(self._train_dataset.lower_bound))
        glog.info("Max duration: {}".format(self._train_dataset.high_bound))

    # Batch_size from 40/101/341
    @parameterized.expand([(40,), (101,), (341,)])
    def test_train_dataset(self, batch_size):
        glog.info("Unittest of train_dataset, fbank and subword specified.")
        count = 0
        dataloader = DataLoader(dataset=self._train_dataset,
                                batch_size=batch_size,
                                collate_fn=ssl_collate_fn)
        for i, batch in enumerate(dataloader):
            count += 1
            glog.info("raw_feat: {}".format(batch["raw_feat"].shape))
            glog.info("auged_feat: {}".format(batch["auged_feat"].shape))
            glog.info("feat_length: {}".format(batch["feat_length"]))
            self.assertEqual(batch["raw_feat"].shape, batch["auged_feat"].shape)
            glog.info(count)

    # Batch_size from 40/101/341
    @parameterized.expand([(40,), (101,), (341,)])
    def test_eval_dataset(self, batch_size):
        glog.info("Unittest of eval_dataset, fbank and subword specified.")
        count = 0
        dataloader = DataLoader(dataset=self._eval_dataset,
                                batch_size=batch_size,
                                collate_fn=ssl_collate_fn)
        for i, batch in enumerate(dataloader):
            count += 1
            glog.info("raw_feat: {}".format(batch["raw_feat"].shape))
            glog.info("auged_feat: {}".format(batch["auged_feat"].shape))
            glog.info("feat_length: {}".format(batch["feat_length"]))
            self.assertEqual(batch["raw_feat"].shape, batch["auged_feat"].shape)
            glog.info(count)


class TestLmDataset(unittest.TestCase):
    """ Unittest of Lm Dataset """

    def setUp(self) -> None:
        self._tokenizer_config = {
            "type": "subword",
            "config": {
                "spm_model": "sample_data/spm/tokenizer.model",
                "spm_vocab": "sample_data/spm/tokenizer.vocab"
            }
        }

        self._dataset_config = {
            "train_data": "sample_data/asr_train_data.json",
            "eval_data": "sample_data/asr_eval_data.json",
            "token_min_filter": 1,
            "token_max_filter": 200,
            "batch_size": 256,
        }
        self._tokenzier = TokenizerSetup(self._tokenizer_config)
        self._train_dataset = LmDataset(
            dataset_json=self._dataset_config["train_data"],
            token_min_filter=self._dataset_config["token_min_filter"],
            token_max_filter=self._dataset_config["token_max_filter"],
            tokenizer=self._tokenzier)

    def test_dataset_info(self):
        glog.info("Total num tokens: {}".format(
            self._train_dataset.total_data_amount))
        glog.info("Min token num: {}".format(self._train_dataset.lower_bound))
        glog.info("Max token num: {}".format(self._train_dataset.high_bound))

    # Batch_size from 40/101/341
    @parameterized.expand([(40,), (101,), (341,)])
    def test_train_dataset(self, batch_size):
        glog.info("Unittest of train_dataset, fbank and subword specified.")
        count = 0
        dataloader = DataLoader(dataset=self._train_dataset,
                                batch_size=batch_size,
                                collate_fn=lm_collate_fn)
        for i, batch in enumerate(dataloader):
            count += 1
            glog.info("text: {}".format(batch["text"].shape))
            glog.info("text_length: {}".format(batch["text_length"]))
            glog.info(count)


if __name__ == "__main__":
    unittest.main()
