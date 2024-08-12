# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.11
""" Cross-Entropy loss impl, Masking strategy enabled. """

import dataclasses
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.functions.masking import make_non_pad_mask


@dataclasses.dataclass
class MaskedCELossConfig:
    """ Config of MaskedCELoss. """
    num_classes: int = 1025  # Codebook size + 1, mask_id = 0 should be taken into account.
    scale_factor: float = 1.0  # Scaling CE loss up
    label_smoothing: float = 0.0


class MaskedCELoss(nn.Module):
    """ Cross-Entropy with mask, mainly for classification task 
        within self-supervised learning. 
    """

    def __init__(self, config: MaskedCELossConfig):
        super(MaskedCELoss, self).__init__()

        # Specify ignore_index as mask, which is 0.
        self._num_classes = config.num_classes
        self._scale_factor = config.scale_factor
        self._label_smoothing = config.label_smoothing

        self._ce_loss = nn.CrossEntropyLoss(
            reduction="none", label_smoothing=self._label_smoothing)

    def forward(self,
                logits: torch.Tensor,
                ori_labels: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """ Cross Entropy loss forward.
            Args:
                logits: Output from encoder, (B, T, D)
                ori_labels: SSL labels from original features, (B, T)
                mask: Mask on labels to predict in ssl task, (B, T) / (B)
            Return:
                loss: Cross Entropy loss, scalar.
        """
        # Encoder output dim equal to Codebook size + 1 since ssl
        # label start from 1 not 0, where 0 indicating mask_id
        max_seq_len = logits.size(1)

        logits = logits.contiguous().reshape(
            -1, self._num_classes)  # (B * T, num_classes)
        logits *= self._scale_factor

        valid_labels = ori_labels.contiguous().reshape(-1)
        loss = self._ce_loss(logits, valid_labels)  # (B, T)

        if mask is not None:
            if len(mask.shape) == 1:
                assert torch.max(mask) == max_seq_len
                mask = make_non_pad_mask(mask).long()
            mask = mask.contiguous().reshape(-1)
            loss *= mask  # Mask out labels not to be predicted
            loss = torch.div(loss.sum(), mask.sum())

        return loss.mean()

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """ Predict interface for metrics evaluation
            Args:
                enc_out: Output of Encoder, (B, T, D)
            Return:
                probs: Probs of each categories, (B, T, num_classes)
        """
        logits *= self._scale_factor
        probs = logits.softmax(dim=-1)
        return probs
