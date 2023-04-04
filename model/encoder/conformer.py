# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.01
""" Conformer Encoder Impl based on torchaudio """

import dataclasses
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


@dataclasses.dataclass
class ConformerConfig:
    """ torchaudio.models.Conformer Config API """
    feats_dim: int = 80
    subsampling_rate: int = 4
    input_dim: int = 512
    num_heads: int = 8
    ffn_dim: int = 2048
    num_layers: int = 8
    depthwise_conv_kernel_size: int = 31
    dropout: float = 0.0
    use_group_norm: bool = False
    convolution_first: bool = False
    output_dim: int = 45  # Num of tokens if CTC


class Subsampling(nn.Module):
    """ NOTE: Subsampling module before Conformer module, since Subsampling 
        lack in torchaudio Conformer Impl. This part is mainly modified from 
        wenet/transformer/subsampling.py excluding position_encoding since
        FAIR team seems not standing by position encoding from vanilla 
        Transformer or Conformer, base on https://arxiv.org/pdf/1904.11660.pdf.
        They held the view pretty much like Conv Layer has already modeling 
        the position why don't we just leave it to em? So, you can find nothing 
        but ConvModules and Attention in this Conformer Impl.
    """

    def __init__(self, idim, odim, subsampling_rate=4):
        super(Subsampling, self).__init__()
        # Add BatchNorm before model to dynamically normalize training data
        self._batchnorm = nn.BatchNorm1d(num_features=idim)

        if subsampling_rate == 4:
            # Subsampling rate = 4
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(1, odim, 3, 2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(odim, odim, 3, 2),
                torch.nn.ReLU(),
            )
            self.linear = torch.nn.Sequential(
                torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
            self.subsampled_length = self._subsampled_length_4

        elif subsampling_rate == 6:
            # Subsampling rate = 6
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(1, odim, 3, 2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(odim, odim, 5, 3),
                torch.nn.ReLU(),
            )
            self.linear = torch.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3),
                                          odim)
            self.subsampled_length = self._subsampled_length_6

        elif subsampling_rate == 8:
            # Subsampling rate = 8
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(1, odim, 3, 2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(odim, odim, 3, 2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(odim, odim, 3, 2),
                torch.nn.ReLU(),
            )
            self.linear = torch.nn.Linear(
                odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim)
            self.subsampled_length = self._subsampled_length_8

    def _subsampled_length_4(self, length: torch.Tensor):
        # Seperate as independent func to comply with torchscript model export
        # avoiding `¡felse`` structure of model for subsampling.
        return torch.div(torch.div(length - 1, 2, rounding_mode="floor") - 1,
                         2,
                         rounding_mode="floor")

    def _subsampled_length_6(self, length: torch.Tensor):
        # Same as above
        return torch.div(torch.div(length - 1, 2, rounding_mode="floor") - 2,
                         3,
                         rounding_mode="floor")

    def _subsampled_length_8(self, length: torch.Tensor):
        # Same as above
        return torch.div(
            torch.div(torch.div(length - 1, 2, rounding_mode="floor") - 1,
                      2,
                      rounding_mode="floor") - 1,
            2,
            rounding_mode="floor")

    def _lengths_to_padding_mask(self, lengths: torch.Tensor) -> torch.Tensor:
        batch_size = lengths.shape[0]
        max_length = int(torch.max(lengths).item())
        padding_mask = torch.arange(
            max_length, device=lengths.device, dtype=lengths.dtype).expand(
                batch_size, max_length) >= lengths.unsqueeze(1)
        return padding_mask

    def forward(self, x: torch.Tensor, length: torch.Tensor):
        # Training graph
        x = self._batchnorm(x.transpose(1, 2)).transpose(1, 2)
        x = x.unsqueeze(1)  # (B, C, T, F)
        x = self.conv(x)
        b, c, t, f = x.size()
        out = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))
        length = self.subsampled_length(length=length)

        # Masked padded residual
        mask = self._lengths_to_padding_mask(length).unsqueeze(-1).repeat(
            1, 1, out.shape[-1])
        out.masked_fill_(mask, 0)

        return out, length

    @torch.inference_mode(mode=True)
    def inference(self, x: torch.Tensor):
        # Inference graph, excluding length output
        x = self._batchnorm(x.transpose(1, 2)).transpose(1, 2)
        x = x.unsqueeze(1)  # (B, C, T, F)
        x = self.conv(x)
        b, c, t, f = x.size()
        out = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return out


class Conformer(nn.Module):
    """ Conformer Impl based on torchaudio, this Conformer
        support non-streaming only.
    """

    def __init__(self, config: ConformerConfig) -> None:
        super(Conformer, self).__init__()
        # Load config
        self._feats_dim = config.feats_dim
        self._subsampling_rate = config.subsampling_rate
        self._input_dim = config.input_dim
        self._num_heads = config.num_heads
        self._ffn_dim = config.ffn_dim
        self._num_layers = config.num_layers
        self._depthwise_conv_kernel_size = config.depthwise_conv_kernel_size
        self._dropout = config.dropout
        self._use_group_norm = config.use_group_norm
        self._convolution_first = config.convolution_first
        self._output_dim = config.output_dim

        # Initialize Subsampling Module
        self._subsampling_module = Subsampling(
            idim=self._feats_dim,
            odim=self._input_dim,
            subsampling_rate=self._subsampling_rate)

        # Initialize Conformer Module
        self._conformer_module = torchaudio.models.Conformer(
            input_dim=self._input_dim,
            num_heads=self._num_heads,
            ffn_dim=self._ffn_dim,
            num_layers=self._num_layers,
            depthwise_conv_kernel_size=self._depthwise_conv_kernel_size,
            dropout=self._dropout,
            use_group_norm=self._use_group_norm,
            convolution_first=self._convolution_first)

        # Initialize output layer for logits output
        self._output_layer = nn.Conv1d(in_channels=self._input_dim,
                                       out_channels=self._output_dim,
                                       kernel_size=1,
                                       bias=True)

    def forward(self, feats: torch.Tensor,
                lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Training graph.
        output, lengths = self._subsampling_module(feats, lengths)
        output, lengths = self._conformer_module(output, lengths)
        output = output.transpose(1, 2)  # (B, T, Dim) -> (B, dim, T)
        logits = self._output_layer(output).transpose(1, 2)

        return logits, lengths

    @torch.inference_mode(mode=True)
    def non_streaming_inference(self, feats: torch.Tensor) -> torch.Tensor:
        # Inference graph, length of output is excluded.
        output = self._subsampling_module.inference(feats)
        # Intuitively, only BatchSize = 1 support during inference, generate
        # dummy_length needed in conformer_module forward.
        dummy_length = torch.Tensor([output.shape[1]]).long().repeat(
            output.shape[0]).to(output.device)
        output, _ = self._conformer_module(output, dummy_length)
        output = output.transpose(1, 2)  # (B, T, Dim) -> (B, dim, T)
        logits = self._output_layer(output)
        logits = F.log_softmax(logits.transpose(1, 2), dim=-1)

        return logits
