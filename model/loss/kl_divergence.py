# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.11
""" KL-Divergence loss with mask, which more suitable for unsupervised
    task accordingly.
"""

import dataclasses
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.functions.masking import make_non_pad_mask


@dataclasses.dataclass
class MaskedKLDivergenceConfig:
    """ Config of MaskedKLDivergence. """
    num_classes: int = 1025  # Codebook size + 1, mask_id = 0 should be taken into account.
    scale_factor: float = 1.0  # Scaling loss up
    label_smoothing: float = 0.0


class MaskedKLDivergence(nn.Module):
    """ KL-Divergence with Mask """

    def __init__(self, config: MaskedKLDivergenceConfig) -> None:
        super(MaskedKLDivergence, self).__init__()

        self._num_classes = config.num_classes
        self._scale_factor = config.scale_factor
        self._label_smoothing = config.label_smoothing

        self._kl_div = nn.KLDivLoss(reduction="none")

    def forward(self,
                logits: torch.Tensor,
                ori_labels: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """ KL-Divergence loss forward.
            Args:
                logits: Output from encoder, (B, T, D)
                ori_labels: SSL labels from original features, (B, T)
                mask: Mask on labels to predict in ssl task, (B) / (B, T) where 1 is valid,
                    vice versa. 
            Return:
                loss: KL-Divergence loss, scalar.
        """
        if mask is not None:
            if len(mask.shape) == 1:
                assert torch.max(mask) == logits.size(1)
                mask = make_non_pad_mask(mask)
            mask = mask.contiguous().reshape(-1)
        else:
            mask = torch.ones_like(ori_labels)
            mask = mask.contiguous().reshape(-1)

        # Encoder output dim equal to Codebook size + 1 since ssl
        # label start from 1 not 0, where 0 indicating mask_id
        logits = logits.contiguous().reshape(-1, self._num_classes)
        logits *= self._scale_factor

        # Build smoothed label
        ori_labels = ori_labels.contiguous().reshape(-1)

        smoothed_label = torch.zeros_like(logits)
        smoothed_label = smoothed_label.fill_(self._label_smoothing /
                                              (self._num_classes - 1))
        confidence = 1 - self._label_smoothing
        smoothed_label.scatter_(-1, ori_labels.unsqueeze(-1), confidence)

        loss = self._kl_div(logits.log_softmax(dim=-1), smoothed_label)
        loss.masked_fill_(~(mask.unsqueeze(-1).bool()),
                          0)  # Mask out padding loss

        return loss.sum() / mask.sum()

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """ Predict interface for metrics evaluation
            Args:
                logits: Output of Encoder, (B, T, D)
            Return:
                probs: Probs of each categories, (B, T, num_classes)
        """
        logits *= self._scale_factor
        probs = logits.log_softmax(dim=-1)
        return probs
