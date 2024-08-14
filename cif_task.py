# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.15
""" CIF ASR task impl. """

import copy
import glog
import torch
import torch.distributed as dist
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import wrap

from dataset.utils import TokenizerSetup
from dataset.frontend.frontend import FeatType
from dataset.dataset import AsrEvalDataset, AsrTrainDataset, asr_collate_fn
from dataset.sampler import DynamicBucketBatchSampler
from model.layer.global_cmvn import GlobalCmvnLayer
from model.encoder.encoder import Encoder
from model.cif.cif_layer import CifLayer, CifLayerConfig
from model.decoder.decoder import Decoder
from model.loss.loss import Loss
from model.utils import AsrMetric, AsrMetricConfig
from optimizer.optim_setup import OptimSetup