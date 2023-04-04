# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.03
""" Encoder Factory """

import torch
import torch.nn as nn

from model.encoder.conformer import Conformer, ConformerConfig
from model.encoder.emformer import Emformer, EmformerConfig
from model.encoder.wav2vec2 import Wav2Vec2CustomizedConfig, Wav2Vec2Encoder


class Encoder(nn.Module):
    """ Encoder interface, accessible for all desgined encoder 
    """

    def __init__(self, config) -> None:
        super(Encoder, self).__init__()

        if config["model"] == "Wav2Vec2":
            self.encoder = Wav2Vec2Encoder(config=Wav2Vec2CustomizedConfig(
                **config["config"]))
        elif config["model"] == "Conformer":
            self.encoder = Conformer(config=ConformerConfig(**config["config"]))
        elif config["model"] == "Emformer":
            self.encoder = Emformer(config=EmformerConfig(**config["config"]))
        # TODO: Supporting other models

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """ Training graph interface """
        # Encoder will foward logits only, which is output right before log softmax
        return self.encoder(x, lengths)

    # Inference model set at the encoder top interface
    @torch.inference_mode(mode=True)
    def non_streaming_inference(self, x: torch.Tensor):
        """ Inference graph interface, Non-streaming """
        return self.encoder.non_streaming_inference(x)

    @torch.inference_mode(mode=True)
    def simu_streaming_inference(self, x: torch.Tensor, config=None):
        """ Inference graph interface, simulated streaming """
        if hasattr(self.encoder, "simu_streaming_inference"):
            # Config of streaming inference should be provided
            return self.encoder.simu_streaming_inference(x, config)
        else:
            raise NotImplementedError
