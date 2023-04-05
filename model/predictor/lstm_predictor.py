# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.05
""" LSTM based predictor, from torchaudio """

import dataclasses
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchaudio.models.rnnt import _Predictor
from typing import List, Optional, Tuple


@dataclasses.dataclass
class LstmPredictorConfig:
    """ Predictor Config """
    num_symbols: int = 128  # <blank_id> = 0, <sos> = num_symbols - 1 are taken into account
    output_dim: int = 1024
    symbol_embedding_dim: int = 512
    num_lstm_layers: int = 3
    lstm_hidden_dim: int = 512
    lstm_layer_norm: bool = True
    lstm_layer_norm_epsilon: float = 1e-3
    lstm_dropout: float = 0.3


class LstmPredictor(nn.Module):
    """ LSTM Predictor wrapped of torchaudio """

    def __init__(self, config: LstmPredictorConfig) -> None:
        super(LstmPredictor, self).__init__()
        # NOTE: Start of sentence token <sos>. Last token of token list self.
        self._sos_token = config.num_symbols - 1
        self._blank_token = 0  # 0 is strictly set for both Ctc and Rnnt.

        self._predictor = _Predictor(
            num_symbols=config.num_symbols,
            output_dim=config.output_dim,
            symbol_embedding_dim=config.symbol_embedding_dim,
            num_lstm_layers=config.num_lstm_layers,
            lstm_hidden_dim=config.lstm_hidden_dim,
            lstm_layer_norm=config.lstm_layer_norm,
            lstm_layer_norm_epsilon=config.lstm_layer_norm_epsilon,
            lstm_dropout=config.lstm_dropout)

    @property
    def sos_token(self) -> int:
        return self._sos_token

    @property
    def blank_token(self) -> int:
        return self._blank_token

    def _left_padding(self, x: torch.Tensor) -> torch.Tensor:
        # left padding input tokens with <blank_id>, only when training.
        # TODO: Future can use <sos> and <eos> to modeling the completeness
        # of Sentence to get preciser endpoint where left_padding <sos> and
        # right padding <eos>
        assert len(x.shape) == 2  # (B, U)
        return F.pad(x.float(), (1, 0, 0, 0), value=float(self.blank_token)).to(
            torch.int32)  # (B, 1 + U)

    def forward(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        state: List[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        # Training graph, beware that left pad input with 1 to <blank_id> (or
        # <sos> in the future) for auto-agressive.
        # input: (B, U); length (B)
        # input should be left padding for rnnt_loss during training
        input = self._left_padding(input)

        if len(state) == 0:
            # Empty [] indicating init state
            output, out_lengths, state_out = self._predictor(
                input, lengths, None)
        else:
            output, out_lengths, state_out = self._predictor(
                input, lengths, state)

        return output, out_lengths, state_out

    @torch.jit.export
    def init_state(self):
        # Initialize state when Asr Session start
        return []

    @torch.jit.export
    @torch.inference_mode(mode=True)
    def streaming_step(
        self, input: torch.Tensor, state: List[List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        # Streaming inference step of Predictor, accept 1 token input only,
        # auto-aggressive, you gotcha.
        assert input.shape[0] == 1  # only support batchsize = 1
        assert input.shape[1] == 1  # only support sequence length = 1

        lengths = torch.tensor([input.shape[1]
                               ]).to(input.device)  # Create dummy length
        if len(state) == 0:
            # Empty [] indicating init state
            output, _, state_out = self._predictor(input, lengths, None)
        else:
            output, _, state_out = self._predictor(input, lengths, state)

        return output, state_out
