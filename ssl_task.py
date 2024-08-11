# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.11
""" Self-supervised Learning task impl. """

import abc
import copy
import glog
import torch
import torch.distributed as dist
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import wrap
from typing import List, Dict

from dataset.frontend.frontend import FeatType
from dataset.dataset import SslTrainDataset, SslEvalDataset, ssl_collate_fn
from dataset.sampler import DynamicBucketBatchSampler
from model.layer.global_cmvn import GlobalCmvnLayer
from model.ssl.best_rq import BestRQLayer, BestRQLayerConfig, MaskingStrategyConfig
from model.encoder.encoder import Encoder
from model.decoder.decoder import Decoder  # Using decoder as final logits layer
from model.loss.loss import Loss
from model.utils import SslMetric, SslMetricConfig
from optimizer.optim_setup import OptimSetup


class SslTask(pl.LightningModule):
    """ Self-supervised learning task. """

    def __init__(self, config) -> None:
        super(SslTask, self).__init__()
