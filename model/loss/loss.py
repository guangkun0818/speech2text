# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.09
""" Loss factory """

import torch
import torch.nn as nn

from typing import Dict

from model.loss.ctc_loss import CtcLoss, CtcLossConfig
from model.loss.rnnt_loss import RnntLoss, RnntLossConfig
from model.loss.pruned_rnnt_loss import PrunedRnntLoss, PrunedRnntLossConfig


class Loss(nn.Module):
    """ Loss interface for all desgined losses """

    def __init__(self, config) -> None:
        super(Loss, self).__init__()

        if config["model"] == "CTC":
            self.loss = CtcLoss(config=CtcLossConfig(**config["config"]))
        elif config["model"] == "Rnnt":
            self.loss = RnntLoss(config=RnntLossConfig(**config["config"]))
        elif config["model"] == "Pruned_Rnnt":
            self.loss = PrunedRnntLoss(config=PrunedRnntLossConfig(
                **config["config"]))

    def forward(self, batch: Dict[str, torch.Tensor]):
        """ Loss training graph """

        loss = self.loss(**batch)
        return loss
