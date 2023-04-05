# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.05
""" Joiner of Rnnt """

import dataclasses
import torch
import fast_rnnt
import torch.nn as nn

from typing import List, Tuple


@dataclasses.dataclass
class JoinerConfig:
    """ Joiner Config interface """
    input_dim: int  # source and target input dimension.
    output_dim: int  # output dimension.
    activation: str = "relu"  # activation func, choose from ("relu", "tanh")
    prune_range: int = 5  # specify as -1 if pruned rnnt loss not applied


class Joiner(nn.Module):
    """ Joiner of both Predictor and Encoder of Rnnt """

    def __init__(self, config: JoinerConfig) -> None:
        super(Joiner, self).__init__()

        # Out_dim of Both Encoder and predictor shall same dim.
        # Basically joiner is just composed with 1 1ayer Linear + activation.
        self._linear = nn.Linear(config.input_dim, config.output_dim, bias=True)
        if config.activation == "relu":
            self._activation = nn.ReLU()
        elif config.activation == "tanh":
            self._activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation {config.activation}")

        self._log_softmax = nn.LogSoftmax(dim=-1)

        self._blank_token = 0  # 0 is strictly set for both Ctc and Rnnt.
        self._prune_range = config.prune_range

    @property
    def prune_range(self) -> int:
        return self._prune_range

    @property
    def blank_token(self) -> int:
        return self._blank_token

    def _do_rnnt_prune(
            self, encoder_out: torch.Tensor, encoder_out_lengths: torch.Tensor,
            predict_out: torch.Tensor, target_lengths: torch.Tensor,
            target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Pruned LM, AM hidden states for PruneRnntLoss 
            Args:
                encoder_out: (B, T, D) 
                encoder_out_lengths: (B) 
                predict_out: (B, 1 + U, D)
                target_lengths: (B)
                target: (B, U) Ground truth of each utterance within batch
            return:
                am_pruned, lm_pruned, boundary, ranges
        """

        boundary = torch.zeros((encoder_out.size(0), 4),
                               dtype=torch.int64,
                               device=encoder_out.device)
        boundary[:, 2] = target_lengths
        boundary[:, 3] = encoder_out_lengths
        assert len(target.shape) == 2  # (B, U)

        _, (px_grad, py_grad) = fast_rnnt.rnnt_loss_smoothed(
            lm=predict_out,
            am=encoder_out,
            symbols=target,
            termination_symbol=self.blank_token,
            lm_only_scale=0.0,
            am_only_scale=0.0,
            boundary=boundary,
            reduction="sum",
            return_grad=True,
        )

        ranges = fast_rnnt.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=self.prune_range,
        )

        # am_pruned : [B, T, prune_range, C]
        # Im_pruned : [B, T, prune_range, C]
        am_pruned, lm_pruned = fast_rnnt.do_rnnt_pruning(am=encoder_out,
                                                         lm=predict_out,
                                                         ranges=ranges)
        return am_pruned, lm_pruned, boundary, ranges

    @torch.jit.unused
    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lengths: torch.Tensor,
        predict_out: torch.Tensor,
        target_lengths: torch.Tensor,
        target: torch.Tensor = torch.empty(0, 0)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ NOTE: U + 1 of target_encodings dims mean raw targets with 
            shape (B, U) should be preppended as (B, 1 + U, D) with 
            <blank_id> (Future maybe <sos>?), this part is pre-processed 
            within Predictor of Rnnt.
            Args:
                encoder_out: (B, T, D) 
                encoder_out_lengths: (B)
                predict_out: (B, U + 1, D) 
                target_lengths: (B)
                target: (B, U) or Empty tensor as default, this will be 
                        utilized when prune rnnt loss applied.
        """
        if self.prune_range > 0:
            assert target.shape[0] == target_lengths.shape[0]
            # Prune with given prune range, return [B, T, prune_range, C]
            encoder_out, predict_out, boundary, ranges = self._do_rnnt_prune(
                encoder_out=encoder_out,
                encoder_out_lengths=encoder_out_lengths,
                predict_out=predict_out,
                target_lengths=target_lengths,
                target=target)
        else:
            # For API consistency
            boundary = None
            ranges = None

        assert len(encoder_out.shape) == 3 or len(encoder_out.shape) == 4
        if len(encoder_out.shape) == 3:
            # Indicating unpruned, unsqueeze for broadcasting
            encoder_out = encoder_out.unsqueeze(2).contiguous()
        assert len(predict_out.shape) == 3 or len(predict_out.shape) == 4
        if len(predict_out.shape) == 3:
            # Indicating unpruned, unsqueeze for broadcasting
            predict_out = predict_out.unsqueeze(1).contiguous()

        joint_encodings = encoder_out + predict_out
        activation_out = self._activation(joint_encodings)
        output = self._linear(activation_out)

        # Use raw output of joiner for rnnt_loss compute since log_softmax will be
        # done within rnnt_loss.
        return output, boundary, ranges

    @torch.jit.export
    @torch.inference_mode(mode=True)
    def streaming_step(self, encoder_out: torch.Tensor,
                       predictor_out: torch.Tensor):
        # Streaming inference step, accept only 1 frame from encoder and 1 token from
        # predictor to predict next. strictly follow Auto-aggressively strategy
        # encoder_out: (1, 1, D), predictor_out: (1, 1, D).
        assert encoder_out.shape[0] == 1 and encoder_out.shape[1] == 1
        assert predictor_out.shape[0] == 1 and predictor_out.shape[1] == 1

        encoder_out = encoder_out.unsqueeze(2).contiguous()
        predictor_out = predictor_out.unsqueeze(1).contiguous()

        joint_encodings = encoder_out + predictor_out
        activation_out = self._activation(joint_encodings)
        output = self._linear(activation_out)

        output = self._log_softmax(output)  #(1, 1, 1, D)
        output = output.squeeze(0).squeeze(0)  #（1,D)
        return output  # Return next token (1, D)
