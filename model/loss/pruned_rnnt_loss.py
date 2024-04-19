# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.09
""" Pruned Rnn-T loss, based on Daniel impl 
    https://github.com/danpovey/fast_rnnt.git
"""

import dataclasses
import torch
import torch.nn as nn
import k2


@dataclasses.dataclass
class PrunedRnntLossConfig:
    """ Pruned Rnnt Loss Config """
    termination_symbol: int = 0  # <blank id> = 0
    rnnt_type: str = "regular"
    delay_penalty: float = 0.0
    reduction: str = "mean"


class PrunedRnntLoss(nn.Module):
    """ Pruned Rnnt Loss Impl """

    def __init__(self, config: PrunedRnntLossConfig) -> None:
        super(PrunedRnntLoss, self).__init__()

        self._termination_symbol = config.termination_symbol
        self._rnnt_type = config.rnnt_type
        self._delay_penalty = config.delay_penalty
        self._reduction = config.reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                logit_lengths: torch.Tensor, target_lengths: torch.Tensor,
                boundary: torch.Tensor, ranges: torch.Tensor):

        # NOTE: logits should strictly be fp32 even if 16-mixed specified.
        loss = k2.rnnt_loss_pruned(
            logits=logits.to(torch.float32),
            symbols=targets,
            ranges=ranges,
            termination_symbol=self._termination_symbol,
            boundary=boundary,
            rnnt_type=self._rnnt_type,
            delay_penalty=self._delay_penalty,
            reduction=self._reduction,
        )

        return loss
