# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.15
""" Mean Absolute Error Loss """

import dataclasses
import torch
import torch.nn as nn


@dataclasses.dataclass
class MaeLossConfig:
    """ Mean Absolute Error Loss config """
    normalize_length: bool = False


class MaeLoss(nn.Module):
    """ Mean Absolute Error Loss impl """

    def __init__(self, config: MaeLossConfig) -> None:
        super(MaeLoss, self).__init__()

        self._normalize_length = config.normalize_length
        self._criterion = nn.L1Loss(reduction="mean")

    def forward(self, tokens_length: torch.Tensor,
                pre_tokens_length: torch.Tensor):
        loss_norm = tokens_length.size(
            0) if not self._normalize_length else tokens_length.sum().float()
        loss = self._criterion(pre_tokens_length, tokens_length)
        loss = torch.div(loss, loss_norm)
        return loss
