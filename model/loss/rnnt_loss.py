# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.09
""" Rnnt Loss from torchaudio """

import dataclasses
import torch
import torch.nn as nn

from torchaudio.transforms import RNNTLoss


@dataclasses.dataclass
class RnntLossConfig:
    """ Config of RnntLoss """
    blank_label: int = 0  # Blank label index
    clamp: float = -1  # Clamp for gradients
    reduction: str = "mean"  # Specifies the reduction to apply to the output


class RnntLoss(nn.Module):
    """ Ront Loss wrapped """

    def __init__(self, config: RnntLossConfig) -> None:
        super(RnntLoss, self).__init__()

        self._rnnt_loss = RNNTLoss(blank=config.blank_label,
                                   clamp=config.clamp,
                                   reduction=config.reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                logit_lengths: torch.Tensor, target_lengths: torch.Tensor):
        """ NOTE: RNNTLoss from 0.12.0 torchaudio did log_softmax within itself.
            Beware that raw targets should be expanded as (B, 1 + U) with 1 left 
            padding dim.
            Args:
                logits: (Batch, max_logit_length, 1 + max_target_length, num_tokens) 
                targets: (Batch, max_target_length) 
                logit_lengths: (batch) 
                target_lengths: (batch)
        """
        loss = self._rnnt_loss(logits, targets.to(torch.int32),
                               logit_lengths.to(torch.int32),
                               target_lengths.to(torch.int32))
        return loss
