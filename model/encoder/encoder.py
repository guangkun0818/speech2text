# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.03
""" Encoder Factory """

import torch
import torch.nn as nn

from model.encoder.conformer import Conformer, ConformerConfig
from model.encoder.emformer import Emformer, EmformerConfig
from model.encoder.wav2vec2 import Wav2Vec2CustomizedConfig, Wav2Vec2Encoder
from model.encoder.zipformer import Zipformer2, Zipformer2Config


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
        elif config["model"] == "Zipformer":
            self.encoder = Zipformer2(config=Zipformer2Config(
                **config["config"]))
        # TODO: Supporting other models

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """ Training graph interface """
        # Encoder will foward logits only, which is output right before log softmax
        return self.encoder(x, lengths)

    @torch.inference_mode(mode=True)
    def streaming_forward(self, x: torch.Tensor, length: torch.Tensor,
                          **config):
        """ Streaming forward interface for inference. """
        if hasattr(self.encoder, "streaming_forward"):
            return self.encoder.streaming_forward(x, length, **config)
        else:
            raise NotImplementedError(
                "{} encoder does not support streaming_forward".format(
                    self.encoder.__class__.__name__))

    def onnx_export(self, export_filename, **config):
        if hasattr(self.encoder, "onnx_export"):
            return self.encoder.onnx_export(export_filename, **config)
        else:
            raise NotImplementedError(
                "{} encoder does not support onnx_export".format(
                    self.encoder.__class__.__name__))
