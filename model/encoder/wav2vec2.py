# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.02
""" Wav2Vec2 Encoder impl based on pretrained model 
    from Huggingface 
    https://huggingface.co/facebook/wav2vec2-base-960h
"""

import dataclasses
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Wav2Vec2Model


@dataclasses.dataclass
class Wav2Vec2CustomizedConfig:
    """ Config of pretrained Wav2vec2.0 encoder """
    pretrained_model: str = "facebook/wav2vec2-large-xlsr-53"
    hidden_size: int = 1024
    label_dim: int = 45


class Wav2Vec2Encoder(nn.Module):
    """ Implement of Wav2Vec2.0 encoder based on pretrained model, 
        please refer to paper https://arxiv.org/pdf/2006.11477.pdf 
        for details.
    """

    def __init__(self, config: Wav2Vec2CustomizedConfig) -> None:
        super(Wav2Vec2Encoder, self).__init__()

        self._pretrained_model = config.pretrained_model
        self._hidden_size = config.hidden_size
        self._label_dim = config.label_dim
        self._wav2vec2_encoder = Wav2Vec2Model.from_pretrained(
            self._pretrained_model)

        # Linear layer to output logits, input_dim = 1024, based on XLRS
        # pretrained model config, and input_dim = 768 for Wav2Vec2-base-960h
        self._linear = nn.Linear(in_features=self._hidden_size,
                                 out_features=self._label_dim)

    def _compute_logits_length(self, lengths: torch.Tensor):
        # Compute lengths of output logits to which will forward to CTC Loss.
        # wav2vec encoder:
        #   -feature_extractor
        #   -feature_projection
        #   -transformer_based_encoder
        # The downsampling only processeed within feature_extractor, therefore,
        # compute final output lengths based on conv layer config of it.
        # Original formula can be found in
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        for module in self._wav2vec2_encoder.feature_extractor.conv_layers:
            conv = module.conv
            # Read config of conv layer
            padding = conv.padding if conv.padding is int else conv.padding[0]
            dilation = conv.dilation if conv.dilation is int else conv.dilation[
                0]
            kernel_size = conv.kernel_size if conv.kernel_size is int else conv.kernel_size[
                0]
            stride = conv.stride if conv.stride is int else conv.stride[0]
            lengths = (lengths + 2 * padding - dilation *
                       (kernel_size - 1) - 1) / stride + 1

        return lengths.long()

    @staticmethod
    def _zero_mean_unit_var_norm(pcms: torch.Tensor,
                                 lengths: torch.Tensor = None):
        # pcms: (B, T); lengths = (B);
        # Data normalization with zero mean and unit var. Basically borrowed
        # from transformers of Huggingface.
        if lengths is not None:
            normed_pcms = []
            for pcm, length in zip(pcms, lengths):
                normed_pcm_slice = (
                    pcm[:length] - pcm[:length].mean(dim=-1, keepdim=True)
                ) / torch.sqrt(pcm[:length].var(dim=-1, keepdim=True) + 1e-7)
                normed_pcms.append(
                    torch.concat([normed_pcm_slice, pcm[length:]], dim=0))
            normed_pcms = torch.vstack(normed_pcms)
        else:
            normed_pcms = (pcms - pcms.mean(dim=-1, keepdim=True)
                          ) / torch.sqrt(pcms.var(dim=-1, keepdim=True) + 1e-7)
        return normed_pcms.to(pcms.device)

    def forward(self, pcms: torch.Tensor, lengths: torch.Tensor):
        """ Training graph """
        pcms = self._zero_mean_unit_var_norm(pcms=pcms, lengths=lengths)
        out_lengths = self._compute_logits_length(lengths=lengths)
        # The pretrained Wav2Vec2 output last_hidden_state and extract_features
        # last_hidden_state supposed to be the logits we need for CTC Loss.
        mask = torch.arange(lengths.max()).unsqueeze(0).to(
            lengths.device) < lengths.unsqueeze(-1)
        output = self._wav2vec2_encoder(pcms,
                                        attention_mask=mask).last_hidden_state
        output = self._linear(output)
        return output, out_lengths

    @torch.inference_mode(mode=True)
    def non_streaming_inference(self, pcms: torch.Tensor):
        """ Inference graph """
        # only output last hidden state as output
        pcms = self._zero_mean_unit_var_norm(pcms=pcms)
        output = self._wav2vec2_encoder(pcms).last_hidden_state
        output = self._linear(output)
        output = F.log_softmax(output, dim=-1)
        return output
