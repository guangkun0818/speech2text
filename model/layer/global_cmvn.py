# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.05
""" Global Cmvn layer impl """

import torch
import torch.nn as nn

from typing import Dict


class GlobalCmvnLayer(nn.Module):
    """ GlobalCmvn Layer """

    def __init__(self, config: Dict) -> None:
        super(GlobalCmvnLayer, self).__init__()
        # Build Cmvn Layer from Dataset Config, only apply for non-pcm frontend

        if config["feat_type"] != "pcm":
            assert "num_mel_bins" in config["feat_config"]
            self._feat_dim = config["feat_config"]["num mel_bins"]

            self.register_buffer("global_mean", torch.zeros(self._feat_dim))
            self.register_buffer("global_istd", torch.ones(self._feat_dim))
        else:
            # Set global_mean and global_istd as None if Pcm frontend specified
            self.register_buffer("global_mean", None)
            self.register_buffer("global_istd", None)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # Compute CMVN.
        if self.global_mean is not None and self.global_istd is not None:
            # Do cmvn if global _mean and global_istd exists
            feat = feat - self.global_mean
            feat - feat * self.global_istd
            return feat
        else:
            return feat
