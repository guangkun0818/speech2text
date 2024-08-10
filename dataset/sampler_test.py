# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.09
""" Unittest of data sampler """

import glog
import unittest

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset.sampler import DynamicBucketBatchSampler
from dataset.dataset import AsrTrainDataset, asr_collate_fn
from dataset.utils import TokenizerSetup


class TestDynamicBucketBatchSampler(unittest.TestCase):
    """ Unittest of DynamicBucketBatchSampler """

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
            "noise_data": "sample_data/noise_data.json",
            "apply_segment": False,
            "dur_min_filter": 0.0,
            "dur_max_filter": 20.0,
            "batch_size": 16,
            "use_bucket_sampler": True,
            "bucket_sampler_config": {
                "num_bucket": 30,
                "key": "duration",
                "min_batch_size": 16,
                "volume_threshold": 800,
            },
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

        self._tokenizer = TokenizerSetup(self._tokenizer_config)
        self._train_dataset = AsrTrainDataset(self._dataset_config,
                                              self._tokenizer)

    def test_distributed_dynamic_bucket_batch_sampler(self):
        sampler = DistributedSampler(self._train_dataset,
                                     num_replicas=1,
                                     rank=0,
                                     shuffle=True,
                                     drop_last=False)
        batch_sampler = DynamicBucketBatchSampler(
            sampler=sampler,
            dataset=self._train_dataset,
            **self._dataset_config["bucket_sampler_config"])
        dataloader = DataLoader(
            dataset=self._train_dataset,
            collate_fn=asr_collate_fn,
            batch_sampler=batch_sampler,
            num_workers=4,
            drop_last=False,
        )

        count = 0
        for i, batch in enumerate(dataloader):
            if i > len(batch_sampler):
                break
            count += 1
            glog.info("feat: {}".format(batch["feat"].shape))
            glog.info("feat_length: {}".format(batch["feat_length"]))
            glog.info("label: {}".format(batch["label"].shape))
            glog.info("label_length: {}".format(batch["label_length"]))
        glog.info(count)


if __name__ == "__main__":
    unittest.main()
