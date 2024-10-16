# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.05
""" Predictor Factory, future support stateless
    predictor or more.
"""

import torch
import torch.nn as nn

from typing import List, Tuple, Optional
from model.predictor.lstm_predictor import (LstmPredictor, LstmPredictorConfig)
from model.predictor.stateless_predictor import (StatelessPredictor,
                                                 StatelessPredictorConfig)


class Predictor(nn.Module):
    """ Predictor interface """

    def __init__(self, config) -> None:
        super(Predictor, self).__init__()

        if config["model"] == "Lstm":
            self.predictor = LstmPredictor(config=LstmPredictorConfig(
                **config["config"]))
        elif config["model"] == "Stateless":
            self.predictor = StatelessPredictor(config=StatelessPredictorConfig(
                **config["config"]))
        else:
            raise NotImplementedError

    def forward(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        state: List[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        # Training graph
        output, out_lengths, state_out = self.predictor(input, lengths, state)
        return output, out_lengths, state_out

    @torch.jit.export
    def init_state(self):
        # Initialize state when Asr Session start.
        return self.predictor.init_state()

    @torch.jit.export
    @torch.inference_mode(mode=True)
    def streaming_step(
        self, input: torch.Tensor, state: List[List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        # Streaming step of inference.
        output, state_out = self.predictor.streaming_step(input, state)

        return output, state_out

    def onnx_export(self, export_path, **config):
        if hasattr(self.predictor, "onnx_export"):
            return self.predictor.onnx_export(export_path, **config)
        else:
            raise NotImplementedError(
                "{} encoder does not support onnx_export".format(
                    self.predictor.__class__.__name__))
