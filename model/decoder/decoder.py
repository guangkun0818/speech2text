# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.05
""" Decoder Factory, supporting AED, 2-pass etc. 
    You name it. 
"""

import torch
import torch.nn as nn

from model.decoder.identity import Identity, IdentityConfig
from model.decoder.projector import Projector, ProjectorConfig


class Decoder(nn.Module):
    """ Decoder interface for all desgined decoders """

    def __init__(self, config) -> None:
        super(Decoder, self).__init__()

        if config["model"] == "Identity":
            self.decoder = Identity(config=IdentityConfig(**config["config"]))
        elif config["model"] == "Projector":
            self.decoder = Projector(config=ProjectorConfig(**config["config"]))
        # TODO: Supporting other models

    def forward(self, x: torch.Tensor, length: torch.Tensor):
        """ Training graph interface """
        return self.decoder(x, length)

    @torch.inference_mode(mode=True)
    def streaming_forward(self, x: torch.Tensor, length: torch.Tensor,
                          **config):
        """ Streaming forward interface for inference. """
        if hasattr(self.decoder, "streaming_forward"):
            return self.decoder.streaming_forward(x, length, **config)
        else:
            raise NotImplementedError(
                "{} decoder does not support streaming_forward".format(
                    self.decoder.__class__.__name__))
