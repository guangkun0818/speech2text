# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.11
""" BEST_RQ self-supervised learning, based on paper
    https://arxiv.org/pdf/2202.01855
"""

import dataclasses
import torch
import random
import math

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Tuple, Optional, Union


@dataclasses.dataclass
class BestRQLayerConfig:
    """ Config of Best-RQ layer """
    pre_post_norm: bool = False  # Using pre_norm and post_norm for CMVN, deprecated in the future
    cnn_kernel_size: Tuple = (
        3, 3)  # Kernel size setting of subsampling module within encoder
    cnn_stride: Tuple = (2, 2
                        )  # Stride setting of subsampling module with encoder
    feat_dim: int = 80  # Acoustic feats dim
    num_codebooks: int = 1  # Specify multi-codebook stablizing training
    codebook_dim: int = 512  # Codebook embedding size
    codebook_size: int = 1024  # Codebook size
    label_basis: str = "euclidean"  # Label generation basis, "euclidean" or "cosine"


@dataclasses.dataclass
class MaskingStrategyConfig:
    """ NOTE: Masking strategy, seperate from Best-RQ layer config. The whole process follow
        each steps as below:
       
        1. Generate masking spans:
        Support different sampling strategy for masking span selection:
            'static': Each masking span is fixed with mask_length
            'uniform': sample from uniform distribution with
                low = mean_span_length - span_length_float_rate
                high = mean_span_length + span_length_float_rate
            'normal': sample from normal distribution with
                loc = mean_span_length
                scale = span_length_float_rate
            'poisson' = sample from possion distribution
        
        2. Randomly scatter generated spans on original feats, where padding parts will be excluded 
        from masking span selection in order to prevent unpredictable performance compromise caused by 
        invalid masking on padding.
            no_overlap: If true, an alternative recursive algorithm will be applied preventing spans selection
                from overlapping.
            min_space: Only works when no_overlap is True, this specify space to keep unmasked between masked spans
    """
    mask_proportion: float = 0.1  # The proportion of mask over the whole feats
    mean_span_length: int = 1  # Mean length of each masking span
    span_select_type: str = "static"  # The way to generate mask spans
    span_length_float_rate: Union[
        int, float,
        None] = None  # Floating rate of each span length (works only for uniform, normal)
    min_num_spans: int = 0  # minimum number of masked spans
    no_overlap: bool = False  # if true, will switch to an alternative recursive algorithm that prevents spans from over lappin
    min_space: int = 0  # only used if no_overlap is True, this is how many elements to keep unmasked between spans
    seed: Optional[int] = None


class BestRQLayer(nn.Module):
    """ Bert-based Random Quantizer layer """

    def __init__(self, layer_config: BestRQLayerConfig,
                 masking_config: MaskingStrategyConfig) -> None:
        super(BestRQLayer, self).__init__()

        # Best_RQ layer config
        self._cnn_kernel_size = layer_config.cnn_kernel_size
        self._cnn_stride = layer_config.cnn_stride

        # Build codebook.
        self._num_codebooks = layer_config.num_codebooks
        self._codebook_dim = layer_config.codebook_dim
        self._codebook_size = layer_config.codebook_size
        assert (
            layer_config.label_basis == "euclidean" or
            layer_config.label_basis == "cosine"
        ), "label_basis shall be chosen from 'euclidean' or 'cosine' exclusively."

        self._label_basis = layer_config.label_basis
        self._codebooks = nn.ParameterList([
            nn.Parameter(data=torch.empty(self._codebook_size,
                                          self._codebook_dim),
                         requires_grad=False)
            for i in range(self._num_codebooks)
        ])

        for i in range(self._num_codebooks):
            nn.init.normal_(self._codebooks[i])

        # matrix for Projection.
        self._feat_dim = layer_config.feat_dim
        self._input_dim = self._feat_dim * math.prod(self._cnn_kernel_size)
        self._projector = nn.Parameter(data=torch.rand(self._input_dim,
                                                       self._codebook_dim),
                                       requires_grad=False)
        nn.init.xavier_normal_(self._projector)

        # Norm right on raw feats
        self._pre_post_norm = layer_config.pre_post_norm
        if self._pre_post_norm:
            self._pre_norm = nn.BatchNorm1d(num_features=self._feat_dim,
                                            affine=False)
            self._post_norm = nn.BatchNorm1d(num_features=self._input_dim,
                                             affine=False)

        # Masking Strategy config
        self._mask_proportion = masking_config.mask_proportion
        self._mean_span_length = masking_config.mean_span_length
        self._span_select_type = masking_config.span_select_type
        self._span_length_float_rate = masking_config.span_length_float_rate
        self._min_num_spans = masking_config.min_num_spans
        self._no_overlap = masking_config.no_overlap
        self._min_space = masking_config.min_space
        self._seed = masking_config.seed

    @property
    def num_codebooks(self):
        # Multi-loss setup for each codebook respectivly
        return self._num_codebooks

    @torch.no_grad()
    def forward(self, raw_feats: torch.Tensor, auged_feats: torch.Tensor,
                length: torch.Tensor) -> Dict[str, torch.Tensor]:
        """ Args:
                raw_feats: Original acoustic feature batch. (B, T, D)
                auged_feats: Augmented acoustic feature batch. (B, T, D)
                length: Actual length of each entry within batch. (B)
            return:
                masked_feats: Masked feature for encoder input. (B, T, D) 
                labels: Quantized label, shape strictly same as encoder output.
                        (B, T // subsampling_rate)
                masked_dim: Masked index of subsampled feats (B, T // subsampling rate) 
                    where 'True' indicates the exact position of label to be predicted.
        """
        if self._pre_post_norm:
            norm_feats = self._pre_norm(raw_feats.transpose(1, 2)).transpose(
                1, 2)  # Do BN on raw feature
            norm_auged_feats = self._pre_norm(auged_feats.transpose(
                1, 2)).transpose(1, 2)  # Do BN on raw feature
        else:
            norm_feats = raw_feats
            norm_auged_feats = auged_feats

        sub_frame_arr, label_lengths, labels = self._get_subsampling_arrangment(
            norm_feats, length)

        masked_feats, masked_dim = self._random_mask(norm_auged_feats,
                                                     sub_frame_arr,
                                                     label_lengths)

        return {
            "masked_feats": masked_feats,
            "labels": labels,
            "masked_dim": masked_dim
        }

    def _get_subsampling_arrangment(self, feats: torch.Tensor,
                                    length: torch.Tensor) -> torch.Tensor:
        """ Internal method to generate Best-RQ ssl labels, frames arrangement on 
            original frame indexs and valid feats lengths after subsampling module 
            within encoder where latter two are mainly used for feats masking strategy.
            
            Args:
                feat: Original acoustic feature batch (B, T, D)
                length: Actual length of each entry within batch. (B)
            return:
                sub_frame_arr: Frame index arrangement after subsampling module 
                    of encoder. (1, T // 4, S)
                    For example if cnn kernel_size: Tuple = (3, 3), cnn_stride: Tuple = (2, 2)
                    applied, returned subsampled_idx would be like:
                        [[[ 0,  1,  2,  3,  4,  5,  6],
                          [ 4,  5,  6,  7,  8,  9, 10],
                          ...
                          [44, 45, 46, 47, 48, 49, 50],
                          [48, 49, 50, 51, 52, 53, 54]]]
                label_lengths: Best-RQ label length, (B)
                labels: Best-RQ Self-supervised label generated on original feats, start from 1 
                    while using 0 as ignore index in CE Loss comput (num_codebooks, B, T // 4)
        """

        B, T, D = feats.size()
        stacked_feats = feats  # feats after subsampling, used for calculating labels
        sub_frame_arr = torch.arange(T).unsqueeze(0).to(
            feats.device
        )  # find the corresponding relation between sub_frame_id and origin_frame_id by this array
        label_lengths = length  # actual length(no padding) after subsampling, (B)

        # Using unfold to simulate Cnn compute in subsampling module with encoder,
        # kinda cumbersome but maybe effective.
        for k, s in zip(self._cnn_kernel_size, self._cnn_stride):
            sub_frame_arr = sub_frame_arr.unfold(1, k, s)
            sub_frame_arr = sub_frame_arr.contiguous().view(
                sub_frame_arr.shape[0], sub_frame_arr.shape[1], -1)
            # According to formula on page below.
            # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            label_lengths = torch.div(
                (label_lengths - k), s, rounding_mode="floor") + 1

            # Stacking original feats following subsampling module arrangement in
            # order to generate ssl labels.
            stacked_feats = stacked_feats.unfold(1, k, s)
            stacked_feats = stacked_feats.contiguous().view(
                stacked_feats.shape[0], stacked_feats.shape[1], -1)

        labels = self._make_label(stacked_feats)
        return sub_frame_arr.unique(dim=-1), label_lengths, labels

    def _random_mask(self, feats: torch.Tensor, sub_frame_arr: torch.Tensor,
                     label_lengths: torch.Tensor) -> torch.Tensor:
        """ Masking strategy on original feats for ssl training.
            
            Args:
                feats: Original acoustic feature batch (B, T, D)
                sub_frame_arr: Frame index arrangement after subsampling module
                    of encoder. (1, T // 4, S)
                label_lengths: Best-RQ label length, (B)
            return:
                masked_feats: Original acoustic feature batch (B, T, D)
                masked_dim: Bool Tensor, masked dim on labels (1 for predict), for 
                    training apparently (B, T // 4)
        """
        # Calculate mask position after downsamping, subsampling mask position = 1, else 0
        batch_size = label_lengths.shape[0]
        sub_timestep = sub_frame_arr.shape[1]
        masked_dim = torch.zeros(batch_size, sub_frame_arr.shape[1]).to(
            feats.device
        )  # Masked frame_id after subsampling, masked position=1 (else=0)
        for batch in range(batch_size):
            tot_sub_timestep = label_lengths[batch].item()
            padding_num = sub_timestep - tot_sub_timestep
            mask_idx = self._compute_mask_indices(sub_timestep, padding_num)
            if mask_idx.numel() != 0:
                masked_dim[batch, mask_idx] = 1
                masked_frame_idx = sub_frame_arr[:, mask_idx].reshape(
                    1, -1
                ).unique(
                    dim=-1
                )  # Obtain original frame idx from sub_frame_arr with masked_idx
                # Mask original feats with random noise according to paper.
                feats[batch, masked_frame_idx, :] = torch.normal(
                    mean=0,
                    std=0.1,
                    size=(1, len(masked_frame_idx),
                          self._feat_dim)).to(dtype=feats.dtype,
                                              device=feats.device)
        return feats, masked_dim

    def _make_label(self, stacked_feats: torch.Tensor):
        """ Make Best-RQ label. select closest one's dim of codebook based 
            on euclidean distance or cosine similarity between codebook and 
            projected feats.
            
            Args:
                stacked_feats: (B, T // 4, self._input_size) stacked feats after subsampling
                    to get make ssl labels
            return:
                labels: SSL labels (num_codebooks, B, T // 4)
        """
        labels = []
        if self._pre_post_norm:
            stacked_feats = self._post_norm(stacked_feats.transpose(
                1, 2)).transpose(1, 2)  # Do post norm after feats stacking
        targets = torch.matmul(stacked_feats, self._projector)

        # Compute labels from multi-codebooks
        for i in range(self._num_codebooks):
            if self._label_basis == "euclidean":
                vector_distances = torch.linalg.vector_norm(
                    F.normalize(targets, dim=-1).unsqueeze(-2) -
                    F.normalize(self._codebooks[i], dim=-1),
                    dim=-1)
                label = torch.argmin(vector_distances,
                                     dim=-1) + 1  # Label start from 1
            elif self._label_basis == "cosine":
                cosine = F.linear(F.normalize(targets, dim=-1),
                                  F.normalize(self._codebooks[i], dim=-1))
                label = torch.argmax(cosine, dim=-1) + 1  # Label start from 1
            labels.append(label)

        # Concat labels into one.
        labels = torch.concat(
            [labels[i].unsqueeze(0) for i in range(len(labels))], dim=0)
        return labels

    def _compute_mask_indices(self, timestep: int,
                              padding_num: Optional[int]) -> torch.Tensor:
        """ Computes random mask spans for a given shape
            Args:
                timestep: the timestep for which to compute masks
                padding_num: Right padding length if original timestep, which help to 
                    prevent masking on padded parts
            return: 
                Selected id on labels to be predict. This will be used to do masking on original feats.
        """
        all_sz = timestep

        all_num_mask = int(
            # add a random number for probabilistic rounding
            self._mask_proportion * all_sz / float(self._mean_span_length) +
            np.random.rand())

        all_num_mask = max(self._min_num_spans, all_num_mask)

        rng = np.random.default_rng(self._seed)

        if padding_num is not None:
            sz = all_sz - padding_num
            num_mask = int(
                # add a random number for probabilistic rounding
                self._mask_proportion * sz / float(self._mean_span_length) +
                rng.random())
            num_mask = max(self._min_num_spans, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        # choose mask length distribution
        if self._span_select_type == "static":
            lengths = np.full(num_mask, self._mean_span_length).tolist()
        elif self._span_select_type == "uniform":
            lengths = rng.integers(
                self._mean_span_length - self._span_length_float_rate,
                self._mean_span_length + self._span_length_float_rate,
                size=num_mask).tolist()
        elif self._span_select_type == "normal":
            lengths = rng.normal(self._mean_span_length,
                                 self._span_length_float_rate,
                                 size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif self._span_select_type == "poisson":
            lengths = rng.poisson(self._mean_span_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection: " + self._span_select_type)

        if sum(lengths) == 0:
            lengths.append(min(self._mean_span_length, sz - 1))

        # Whether masks could be overlapping
        if self._no_overlap:
            # if true, will switch to an alternative recursive algorithm that prevents spans from overlapping
            mask_idc = []

            # this function used to calculate new parts
            def arrange(s, e, length, keep_length):
                if s == e - length:
                    span_start = s
                else:
                    span_start = rng.integers(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))
                new_parts = []
                if span_start - s - self._min_space >= keep_length:
                    new_parts.append((s, span_start - self._min_space + 1))
                if e - span_start - length - self._min_space > keep_length:
                    new_parts.append((span_start + length + self._min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)

            for length in sorted(lengths, reverse=True):
                # Create a new 1-dimensional array from an iterable object,
                # this array size is same as parts's size,
                # the element in this array equal to len if satisfy: len >= length + self. min_space,
                # else equal to 0
                lens = np.fromiter(
                    (e - s if e - s >= length + self._min_space else 0
                     for s, e in parts),
                    np.int64,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                # the longer len, the more probability to choice
                c = rng.choice(len(parts), p=probs)
                # update parts
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            # allow masks to overlap
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = rng.choice(sz - min_len, num_mask, replace=False)
            mask_idc = np.asarray([
                mask_idc[j] + offset
                for j in range(len(mask_idc))
                for offset in range(lengths[j])
            ])
        mask_idc = np.unique(mask_idc[mask_idc < sz])
        return torch.LongTensor(mask_idc).contiguous().view(-1)
