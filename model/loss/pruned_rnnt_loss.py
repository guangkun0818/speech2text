# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.09
""" Pruned Rnn-T loss, based on Daniel impl 
    https://github.com/danpovey/fast_rnnt.git
"""

import dataclasses
import torch
import torch.nn as nn
import fast_rnnt


@dataclasses.dataclass
class PrunedRnntLossConfig:
    """ Pruned Rnnt Loss Config """
    termination_symbol: int = 0  # <blank id> = 0
    reduction: str = "mean"


class PrunedRnntLoss(nn.Module):
    """ Pruned Rnnt Loss Impl """

    def __init__(self, config: PrunedRnntLossConfig) -> None:
        super(PrunedRnntLoss, self).__init__()

        self._termination_symbol = config.termination_symbol
        self._reduction = config.reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                logit_lengths: torch.Tensor, target_lengths: torch.Tensor,
                boundary: torch.Tensor, ranges: torch.Tensor):
        # NOTE: logits should strictly be fp32 even if 16-mixed specified.
        loss = fast_rnnt.rnnt_loss_pruned(
            logits=logits.to(torch.float32),
            symbols=targets,
            ranges=ranges,
            termination_symbol=self._termination_symbol,
            boundary=boundary,
            reduction=self._reduction,
        )
        return loss
