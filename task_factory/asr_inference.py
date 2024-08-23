# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.22
""" Abstract class for different setting asr system inference. 
"""

import os
import abc
import glog
import time
import torch
import torch.distributed as dist
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset.utils import TokenizerSetup
from dataset.frontend.frontend import FeatType
from dataset.dataset import AsrTestDataset, asr_test_collate_fn
from dataset.sampler import DynamicBucketBatchSampler
from model.utils import word_error_rate


class AbcAsrInference(pl.LightningModule):
    """ Base class of asr inference for inherit. """

    def __init__(self, infer_config) -> None:
        super(AbcAsrInference, self).__init__()

        # Setting up export test report
        self._export_path = infer_config["task"]["export_path"]
        curr_time = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
        self._test_report = os.path.join(self._export_path,
                                         "test_report_{}".format(curr_time))

        self._testset_json = infer_config["testset"]["test_data"]
        self._testset_config = infer_config["testset"]["config"]
        self._decoding_config = infer_config["decoding"]

        # For final metric compute
        self._reference = []
        self._predicton = []

    def test_dataloader(self):
        """ Set up eval dataloader. """

        dataset = AsrTestDataset(testset_config=self._testset_config,
                                 testset_json=self._testset_json)
        sampler = DistributedSampler(dataset,
                                     num_replicas=dist.get_world_size(),
                                     rank=dist.get_rank(),
                                     shuffle=False,
                                     drop_last=False)
        dataloader = DataLoader(dataset=dataset,
                                sampler=sampler,
                                collate_fn=asr_test_collate_fn,
                                batch_size=self._testset_config["batch_size"],
                                num_workers=4)
        return dataloader

    @abc.abstractmethod
    def test_step(self, batch, batch_idx):
        ...

    def _export_decoded_results(self, utts, hyps, refs):
        with open(self._test_report, 'a+') as test_report_f:
            for utt, hyp, ref in zip(utts, hyps, refs):
                wer = word_error_rate([hyp], [ref], show_on_screen=False)
                test_report_f.write("utt: {}\n".format(utt))
                test_report_f.write("hyp: {}\n".format(hyp))
                test_report_f.write("ref: {}\n".format(ref))
                test_report_f.write("wer: {:.3f}\n".format(wer * 100))
                test_report_f.write("\n")

    def on_test_end(self) -> None:
        tot_wer = word_error_rate(self._predicton,
                                  self._reference,
                                  show_on_screen=False)
        glog.info("Total WER: {:.3f}".format(tot_wer * 100))

        with open(self._test_report, 'a+') as test_report_f:
            test_report_f.write("Total WER: {:.3f}\n".format(tot_wer * 100))
