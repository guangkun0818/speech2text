# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.05
""" Decoder Factory, supporting AED, 2-pass etc. 
    You name it. 
"""

import torch
import torch.nn as nn

from model.decoder.identity import Identity, IdentityConfig


class Decoder(nn.Module):
    """ Decoder interface for all desgined decoders """

    def __init__(self, config) -> None:
        super(Decoder, self).__init__()

        if config["model"] == "Identity":
            self.decoder = Identity(config=IdentityConfig(**config["config"]))
        # TODO: Supporting other models

    def forward(self, x: torch.Tensor, length: torch.Tensor):
        """ Training graph interface """
        return self.decoder(x, length)

    # Inference model set at the decoder top interface
    @torch.inference_mode(mode=True)
    def non_streaming_inference(self, x: torch.Tensor) -> torch.Tensor:
        """ Inference graph interface, Non-streaming """
        return self.decoder.non_streaming_inference(x)

    @torch.inference_mode(mode=True)
    def simu_streaming_inference(self, x: torch.Tensor, config=None):
        """ Inference graph interface, simulated streaming"""
        if hasattr(self.decoder, "simu_streaming_inference"):
            return self.decoder.simu_streaming_inference(x, config)
        else:
            raise NotImplementedError
