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
        elif config["model"] == "Pruned_Runt":
            self.loss = PrunedRnntLoss(config=PrunedRnntLossConfig(
                **config["config"]))

    def forward(self, batch: Dict[str, torch.Tensor]):
        """ Loss training graph """
        assert "log_probs" in batch
        assert "inputs_length" in batch
        assert "targets" in batch
        assert "targets_length" in batch
        # This shitty desgin is freaking sacifice of consistency API of
        # all different losses.
        if "boundary" in batch.keys() and "ranges" in batch.keys(
        ) and batch["boundary"] is not None and batch["ranges"] is not None:
            # Indicating PrunedntLoss applied
            loss = self.loss(batch["log_probs"], batch["targets"],
                             batch["inputs_length"], batch["targets_length"],
                             batch["boundary"], batch["ranges"])
        else:
            loss = self.loss(batch["log_probs"], batch["targets"],
                             batch["inputs_length"], batch["targets_length"])
        return loss
