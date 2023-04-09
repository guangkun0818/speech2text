# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.05
""" Wrapped CTC loss """

import dataclasses
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class CtcLossConfig:
    """ Config of CTCLoss """
    blank_label: int = 0
    reduction: str = "mean"
    zero_infinity: bool = True


class CtcLoss(nn.Module):
    """ Ctc Loss
        config: CtcLossConfig
    """

    def __init__(self, config: CtcLossConfig):
        super(CtcLoss, self).__init__()

        self._blank_label = config.blank_label
        self._reduction = config.reduction
        self._zero_infinity = config.zero_infinity
        self._loss = nn.CTCLoss(blank=self._blank_label,
                                reduction=self._reduction,
                                zero_infinity=self._zero_infinity)

    def forward(self, enc_out, targets, inputs_length, targets_length):
        # CTC only accept Log_Probs
        log_probs = F.log_softmax(enc_out, dim=-1)
        # Need transpose (B, T, N) as (T, B, N), where B
        # stands for batch_size, T stands for sequence length
        log_probs = log_probs.transpose(0, 1).to(dtype=torch.float32)
        return self._loss(log_probs, targets, inputs_length, targets_length)
