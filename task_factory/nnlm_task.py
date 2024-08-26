# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.04.20
""" Nnlm task impl. """

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