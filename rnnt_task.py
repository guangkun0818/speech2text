# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.04.20
""" Rnnt-arch ASR task impl. """

import copy
import glog
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch.distributed.fsdp.wrap import wrap

from dataset.utils import TokenizerSetup
from dataset.frontend.frontend import KaldiWaveFeature, DummyFrontend
from dataset.dataset import AsrEvalDataset, AsrTrainDataset, asr_collate_fn
from model.layer.global_cmvn import GlobalCmvnLayer
from model.encoder.encoder import Encoder
from model.decoder.decoder import Decoder
from model.predictor.predictor import Predictor
from model.joiner.joiner import Joiner, JoinerConfig
from model.loss.loss import Loss
from model.utils import AsrMetric, AsrMetricConfig


class BaseRnntTask(pl.LightningModule):
    """ Build CTC task """


class RnntTask(BaseRnntTask):
    """ Vanilla Rnnt Task, unpruned rnnt loss. """


class PrunedRnntTask(BaseRnntTask):
    """ k2 Pruned Rnnt task. """