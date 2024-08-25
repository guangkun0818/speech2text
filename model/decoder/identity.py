# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.05
""" Identity Decoder. Well, for compatiblity 
    with CTC task.
"""

import dataclasses
import torch
import torch.nn as nn


@dataclasses.dataclass
class IdentityConfig:
    """ Config of Identity, future inplemented decoder should 
        config itself in this fashion.
    """
    dummy: int = -1


class Identity(nn.Module):

    def __init__(self, config: IdentityConfig):
        super(Identity, self).__init__()

    def forward(self, x: torch.Tensor, length: torch.Tensor):
        """ Training graph interface """
        # Well, literally Identity
        return x, length

    def non_streaming_inference(self, x: torch.Tensor) -> torch.Tensor:
        """ Inference graph interface, Non-streaming """
        return x

    def simu_streaming_inference(self, x, config=None):
        """ Inference graph interface, simulated streaming
            config is just for API compliance.
        """
        return x

    def streaming_forward(self,
                          x: torch.Tensor,
                          length: torch.Tensor,
                          dummy=-1):
        """ NOTE: Streaming forward interface. """
        return x, length
