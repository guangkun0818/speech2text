# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.05
""" Project layer, project output of encoder into designed dim.
"""

import dataclasses
import torch

import torch.nn as nn
from typing import Tuple


@dataclasses.dataclass
class ProjectorConfig:
    """ Config of Projector, future inplemented decoder should 
        config itself in this fashion.
    """
    input_dim: int = 512
    output_dim: int = 1000
    dropout_p: float = 0.1


class Projector(nn.Module):

    def __init__(self, config: ProjectorConfig) -> None:
        super(Projector, self).__init__()

        self._fc = nn.Linear(in_features=config.input_dim,
                             out_features=config.output_dim)
        self._dropout = nn.Dropout(p=config.dropout_p)

    def forward(self, x: torch.Tensor,
                length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Training graph
        output = self._dropout(self._fc(x))
        return output, length

    def streaming_forward(self,
                          x: torch.Tensor,
                          length: torch.Tensor,
                          dummy=-1):
        """ NOTE: Streaming forward interface. """
        output = self._dropout(self._fc(x))
        return output, length

    def non_streaming_inference(self, x: torch.Tensor) -> torch.Tensor:
        """ Inference graph interface, Non-streaming """
        output = self._dropout(self._fc(x))
        return output

    def simu_streaming_inference(self, x, config=None):
        """ Inference graph interface, simulated streaming
            config is just for API compliance.
        """
        output = self._dropout(self._fc(x))
        return output
