# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.27
""" Rnn Lm impl """

import dataclasses
import torch
import torch.nn as nn

from typing import Tuple


@dataclasses.dataclass
class RnnLmConfig:
    """ Rnn lm config """
    num_symbols: int = 128
    symbol_embedding_dim: int = 512
    num_rnn_layer: int = 3
    dropout: float = 0.0
    bidirectional: bool = False


class RnnLm(nn.Module):
    """ Rnn Lm, Embedding + LSTM """

    def __init__(self, config: RnnLmConfig) -> None:
        super(RnnLm, self).__init__()

        self._embedding_dim = config.symbol_embedding_dim
        self._num_symbols = config.num_symbols
        self._embedding = nn.Embedding(num_embeddings=self._num_symbols,
                                       embedding_dim=self._embedding_dim)

        self._num_rnn_layer = config.num_rnn_layer
        self._dropout = config.dropout
        self._bidirectional = config.bidirectional
        self._rnn_layer = nn.LSTM(input_size=self._embedding_dim,
                                  hidden_size=self._embedding_dim,
                                  num_layers=self._num_rnn_layer,
                                  batch_first=True,
                                  dropout=self._dropout,
                                  bidirectional=self._bidirectional)

        self._logits_layer = nn.Linear(in_features=self._embedding_dim,
                                       out_features=self._num_symbols)

    def forward(self, x: torch.Tensor,
                x_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._embedding(x)
        x, _ = self._rnn_layer(x)
        x = self._logits_layer(x)
        return x, x_lens
