# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.27
""" Rnn Lm impl """

import dataclasses
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from model.functions.masking import make_non_pad_mask


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

    def init_states(self, beam_size):
        # Initialize rnn cache when streaming inference start. Batch size = beam_size which
        # corresponding to decoding setting within asr_system.
        # LSTM cache: Tuple[torch.Tensor, torch.Tensor]
        states = (torch.zeros(self._num_rnn_layer, beam_size,
                              self._embedding_dim),
                  torch.zeros(self._num_rnn_layer, beam_size,
                              self._embedding_dim))  # h_0, c_0
        return states

    def forward(self, x: torch.Tensor,
                x_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._embedding(x)
        x, _ = self._rnn_layer(x)
        x = self._logits_layer(x)
        return x, x_lens

    @torch.inference_mode(mode=True)
    def score(self, tokens: torch.Tensor, tokens_length: torch.Tensor):
        """ Score on whole sequence. 
            Args:
                tokens: (B, T) Tokenized text sequence.
                tokens_length: (B) Corresponding valid sequence length.
            Returns:
                scores: (B) Sum of log_probs
        """
        logits, tokens_length = self.forward(tokens, tokens_length)
        log_probs = F.log_softmax(logits, dim=-1)

        tgt_log_probs = log_probs[:, :-1].gather(
            2, tokens[:, 1:].unsqueeze(2)).squeeze(2)  # (B, T)
        tgt_masks = make_non_pad_mask(tokens_length - 1)

        scores = torch.sum(tgt_log_probs * tgt_masks, dim=-1)
        return scores

    @torch.inference_mode(mode=True)
    def score_step(self, tokens: torch.Tensor, states):
        """ Score on coming token within beam. 
            Args:
                tokens: (beam_size)
                states: (h_0, c_0) states within LSTM
            Returns:
                scores: (beam_size, vocab_size) log_probs
        """
        x = self._embedding(tokens.unsqueeze(-1))
        x, states = self._rnn_layer(x, states)
        logits = self._logits_layer(x)
        log_probs = F.log_softmax(logits, dim=-1).squeeze(1)

        return log_probs, states
