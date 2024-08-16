# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.11
""" Continuous Integrate-Fire Layer impl,
    based on https://arxiv.org/pdf/1905.11235, 
    which is core part for Non-autoregressive 
    AED system. 
"""

import torch
import dataclasses
import torch.nn as nn

from model.functions.masking import make_pad_mask


@dataclasses.dataclass
class CifLayerConfig:
    """ Config of Cif Laywer """
    idim: int = 512  # the dim of encoder output
    l_pad: int = 0  # the padding of Conv1d
    r_pad: int = 0
    dropout: float = 0.1
    threshold: float = 1.0
    smooth_factor: float = 1.0
    noise_threshold: float = 0.0
    tail_threshold: float = 0.45


class CifLayer(nn.Module):
    """ Continuous Integrate-Fire Layer """

    def __init__(self, config: CifLayerConfig) -> None:
        super(CifLayer, self).__init__()

        self._pad = nn.ConstantPad1d((config.l_pad, config.r_pad), 0)
        self._cif_conv1d = nn.Conv1d(in_channels=config.idim,
                                     out_channels=config.idim,
                                     kernel_size=config.l_pad + config.r_pad +
                                     1,
                                     groups=config.idim)
        self._cif_output = nn.Linear(config.idim, 1)
        self._dropout = nn.Dropout(p=config.dropout)
        self._threshold = config.threshold
        self._smooth_factor = config.smooth_factor
        self._noise_threshold = config.noise_threshold
        self._tail_threshold = config.tail_threshold

    def forward(self,
                encoder_out: torch.Tensor,
                encoder_out_length: torch.Tensor = None,
                target_label: torch.Tensor = None,
                target_label_length: torch.Tensor = None,
                ignore_id=-1,
                mask_chunk_predictor=None):
        """ Args:
                encoder_out: Encoder output (B, T, D).
                encoder_out_length: Created by encoder_output_length, (B, 1, T), 
                target_label: target token label. 
                target_label_length: target token label length. 
                ignore_id:
                mask_chunk_predictor:
        """
        context = encoder_out.transpose(
            1, 2).contiguous()  # （B, T, D） -> （B, D, T） for Conv1d
        queries = self._pad(context)
        memory = self._cif_conv1d(queries)
        output = memory + context
        output = self._dropout(output)
        output = output.transpose(1, 2).contiguous()
        output = torch.relu(output)
        output = self._cif_output(output)  # (B, T, D) -> (B, T, 1)
        alphas = torch.sigmoid(output)
        alphas = torch.relu(alphas * self._smooth_factor -
                            self._noise_threshold)

        mask = None
        if encoder_out_length is not None:
            mask = (~make_pad_mask(encoder_out_length).unsqueeze(1)).to(
                output.device)
        if mask is not None:
            mask = mask.transpose(
                -1, -2).contiguous().float()  # (B, T, 1) -> (B, 1, T)
            alphas = alphas * mask
        if mask_chunk_predictor is not None:
            alphas = alphas * mask_chunk_predictor
        alphas = alphas.squeeze(-1)  # (B, T, 1) -> (B, T)
        if mask is not None:
            mask = mask.squeeze(-1)  # (B, T, 1) -> (B, T)

        if target_label_length is not None:
            target_length = target_label_length  # target_length: token number true value
        elif target_label is not None:
            target_length = (target_label != ignore_id).float() - sum(-1)
        else:
            target_length = None

        token_num_hat = alphas.sum(
            -1)  # the estimation of token number, token_num _hat: (B)

        if target_length is not None:
            # Scaling strategy in training: the weight a is scaled by target length during training so as to
            # match the number of acoustic embeddings Ea with target embeddings Ec
            alphas *= (target_length / token_num_hat).unsqueeze(1)

        elif self._tail_threshold > 0:  # without target length, for inference
            encoder_out, alphas, token_num_hat = self._tail_process_fn(
                encoder_out, alphas, token_num_hat,
                mask=mask)  # Tail padding one frame, T -> T + 1

        acoustic_embeds, cif_peak = self.continuous_integrate_fire(
            encoder_out, alphas,
            self._threshold)  # Core function, (B, max_label_len, D), (B, T)

        if target_length is None and self._tail_threshold > 0.0:
            token_num_int = torch.max(token_num_hat).type(torch.int32).item()
            acoustic_embeds = acoustic_embeds[:, :token_num_int, :]

        # token_num hat is float if self._tail_threshold == 0.0, it may be a bug
        # token_num_hat = torch. floor (token_num_hat). int()
        return acoustic_embeds, cif_peak, token_num_hat, alphas

    def _tail_process_fn(self,
                         encoder_out: torch.Tensor,
                         alphas: torch.Tensor,
                         token_num=None,
                         mask=None):
        batch, time_stamp, dim = encoder_out.size()
        if mask is not None:
            zeros_t = torch.zeros((batch, 1),
                                  dtype=torch.float32,
                                  device=alphas.device)
            ones_t = torch.ones_like(zeros_t)
            mask_1 = torch.cat([mask, zeros_t], dim=1)
            mask_2 = torch.cat([ones_t, mask], dim=1)
            mask = mask_2 - mask_1  # mask: (1, 1, 1, 0, 0) -> (0, 0, 0, 1, 0, 0)
            tail_threshold = mask * self._tail_threshold
            alphas = torch.cat([alphas, zeros_t], dim=1)
            alphas = torch.add(
                alphas, tail_threshold
            )  # make sure alpha > tail_threshold in the last valid frame
        else:
            tail_threshold = torch.tensor([[self._tail_threshold]],
                                          dtype=alphas.dtype).to(alphas.device)
            tail_threshold = tail_threshold.expand(
                batch, 1)  # Add this line to fix bug
            alphas = torch.cat([alphas, tail_threshold], dim=1)

        zeros = torch.zeros((batch, 1, dim),
                            dtype=encoder_out.dtype).to(encoder_out.device)
        encoder_out = torch.cat([encoder_out, zeros], dim=1)
        token_num = alphas.sum(dim=-1)
        token_num_floor = torch.floor(token_num)

        return encoder_out, alphas, token_num_floor

    @staticmethod
    def continuous_integrate_fire(encoder_out: torch.Tensor,
                                  alphas: torch.Tensor,
                                  threshold: torch.Tensor):
        batch_size, len_time, encoder_out_size = encoder_out.size(
        )  # encoder_out: (B, T, D)

        # loop varss
        integrate = torch.zeros([batch_size],
                                device=encoder_out.device)  # integrate: (B)
        frame = torch.zeros([batch_size, encoder_out_size],
                            device=encoder_out.device)  # frame: (B, D)

        # intermediate vars along time
        list_fires = []
        list_frames = []
        for t in range(len_time):
            alpha_t = alphas[:, t]  # alpha(time t): (B)
            # why torch.ones, not torch. tensor (threshold)???
            distribution_completion = torch.ones(
                [batch_size], device=encoder_out.device) - integrate

            integrate += alpha_t  # (B)
            list_fires.append(integrate)  # list_fires length: T

            fire_place = integrate >= threshold  # (B)
            integrate = torch.where(
                fire_place,
                integrate - torch.ones([batch_size], device=encoder_out.device),
                integrate)  # make sure integrate is not greater than 1
            cur = torch.where(fire_place, distribution_completion,
                              alpha_t)  # the left part of alpha, cur: (B)
            remainds = alpha_t - cur  # the right part of alpha, remainds: (B)

            frame += cur.unsqueeze(1) * encoder_out[:, t, :]  # (B, D)
            list_frames.append(
                frame)  # the num of list_frames equal to t, not sum(alpha)

            frame = torch.where(
                fire_place.unsqueeze(1).repeat(1, encoder_out_size),  # (B, D) 
                remainds.unsqueeze(1) * encoder_out[:, t, :],
                frame)

        fires = torch.stack(list_fires, 1)  # (B, T), integrate of alpha
        frames = torch.stack(list_frames, 1)  # (B, T, D)
        list_ls = []
        # Why torch.round, not torch.floor? Because in training phase using scaling
        # strategy, the sum must be int.
        len_labels = torch.round(alphas.sum(-1)).int()  # (B)
        max_label_len = len_labels.max()

        for b in range(batch_size):
            fire = fires[b, :]  # (T)
            l = torch.index_select(
                frames[b, :, :],  # frames[b, :,:] : (T, D)
                0,
                torch.nonzero(
                    fire >= threshold).squeeze())  # the frames at fire place,
            pad_l = torch.zeros([max_label_len - l.size(0), encoder_out_size],
                                device=encoder_out.device)
            list_ls.append(torch.cat([l, pad_l], 0))  # (max_label_len, D)

        return torch.stack(list_ls, 0), fires  # (B, max_label_len, D), (B, T)
