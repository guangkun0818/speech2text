# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.22
""" Abstract class for different setting asr system inference. 
"""

import abc
import glog
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset.utils import TokenizerSetup
from dataset.frontend.frontend import FeatType
from dataset.dataset import AsrTestDataset, asr_collate_fn
from dataset.sampler import DynamicBucketBatchSampler
from model.utils import word_error_rate


class AbcAsrInference(pl.LightningModule):
    """ Base class of asr inference for inherit. """

    def __init__(self, infer_config) -> None:
        super(AbcAsrInference, self).__init__()

        self._export_path = infer_config["export_path"]
        self._test_data = infer_config["test_data"]
        self._reference = []
        self._predicton = []

    def test_dataloader(self) -> abc.Any:
        ...

    @abc.abstractmethod
    def test_step(self, batch, batch_idx):
        ...

    def on_test_end(self) -> None:
        tot_wer = word_error_rate(self._predicton,
                                  self._reference,
                                  show_on_screen=False)
        glog.info("Total WER: {:.3f}".format(tot_wer * 100))
