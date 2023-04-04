# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.01
""" Emformer Encoder Impl base on torchaudio, based on paper 
    https://arxiv.org/pdf/2010.10759.pdf
"""

import dataclasses
import torch
import torchaudio
import torch.nn as nn

from typing import Optional, Tuple, List

# Borrow subsampling layer from Conformer Encoder
from model.encoder.conformer import Subsampling


@dataclasses.dataclass
class EmformerConfig:
    """ Emformer config API """
    feats_dim: int = 80
    subsampling_rate: int = 4
    infer_chunk_size: int = 20
    emformer_input_dim: int = 512
    num_heads: int = 8
    ffn_dim: int = 2048
    num_layers: int = 20
    segment_length: int = 4
    dropout: float = 0.1
    activation: str = 'gelu'
    left_context_length: int = 30
    right_context_length: int = 0
    max_memory_size: int = 0
    weight_init_scale_strategy: Optional[str] = 'depthwise'
    tanh_on_mem: bool = False
    output_dim: int = 1024


class Emformer(nn.Module):
    """ Custom desgined Emformer, with Subsampling-auged """

    def __init__(self, config: EmformerConfig):
        super(Emformer, self).__init__()

        # Streaming inference related config
        self._infer_chunk_size = config.infer_chunk_size
        self._segment_length = config.segment_length

        # Cache-free when streaming-mode apply to Subsampling layer, trade-off
        # precision with memory and computation.
        self._subsampling_module = Subsampling(
            idim=config.feats_dim,
            odim=config.emformer_input_dim,
            subsampling_rate=config.subsampling_rate)

        # Streaming supported Emformer
        self._emformer_module = torchaudio.models.Emformer(
            input_dim=config.emformer_input_dim,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            num_layers=config.num_layers,
            segment_length=config.segment_length,
            dropout=config.dropout,
            activation=config.activation,
            left_context_length=config.left_context_length,
            right_context_length=config.right_context_length,
            max_memory_size=config.max_memory_size,
            weight_init_scale_strategy=config.weight_init_scale_strategy,
            tanh_on_mem=config.tanh_on_mem,
        )
        # Cache-free streaming, no-worry about it.
        self._output_linear = torch.nn.Linear(config.emformer_input_dim,
                                              config.output_dim)
        self._layer_norm = torch.nn.LayerNorm(config.output_dim)

    def forward(self, feats: torch.Tensor,
                lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Training graph
        # feats: (B, T, D); length: (B)
        output, out_lengths = self._subsampling_module(
            feats, lengths)  # output: (B, T / subsamp_rate, D)
        output, out_lengths = self._emformer_module(
            output, out_lengths)  # output: (B, T / subsamp_rate, D)
        output = self._output_linear(output)
        output = self._layer_norm(output)

        return output, out_lengths

    @torch.jit.export
    def init_state(self):
        # Initialize state when Asr Session start
        return []

    @torch.jit.export
    @torch.inference_mode(mode=True)
    def streaming_step(self, feats: torch.Tensor,
                       states: List[List[torch.Tensor]]):
        # When streaming mode applied, Emformer shall sequentially forward with
        # stream_step when feats flood in.
        output = self._subsampling_module.inference(feats)
        assert feats.shape[1] == self._infer_chunk_size and output.shape[
            1] == self._segment_length, "Please review your config of streaming infer."

        dummy_lengths = torch.tensor([output.shape[1]
                                     ]).to(output.device).to(torch.int32)
        if len(states) == 0:
            # Indicating this is the init status
            output, _, states = self._emformer_module.infer(
                output, dummy_lengths, None)
        else:
            # Intermediate step
            output, _, states = self._emformer_module.infer(
                output, dummy_lengths, states)

        output = self._output_linear(output)
        output = self._layer_norm(output)

        return output, states

    @torch.inference_mode(mode=True)
    def non_streaming_inference(self, x: torch.Tensor):
        # As usual, support batchsize = 1 only. x(1, T, D)
        dummy_lengths = torch.tensor([x.shape[1]]).to(x.device).to(torch.int32)
        output, _ = self.forward(x, dummy_lengths)
        return output
