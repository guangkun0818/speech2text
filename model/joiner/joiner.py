# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.05
""" Joiner of Rnnt """

import os
import dataclasses
import onnx
import torch
import k2
import torch.nn as nn

from typing import List, Tuple


@dataclasses.dataclass
class JoinerConfig:
    """ Joiner Config interface """
    input_dim: int  # Input dimension of encoder_out and predictor_out.
    output_dim: int  # Output dimension, refered as vocab size
    inner_dim: int = 256  # Inner dim of last projection layer
    activation: str = "relu"  # activation func, choose from ("relu", "tanh")
    prune_range: int = 5  # specify as -1 if pruned rnnt loss not applied
    lm_scale: float = 0.0  # lm_scale applied in simple_loss of pruned_rnnt
    am_scale: float = 0.0  # am_scale applied in simple_loss of pruned_rnnt
    use_out_project: bool = True  # If apply last output projection, if false, params saved


class Joiner(nn.Module):
    """ Joiner of both Predictor and Encoder of Rnnt """

    def __init__(self, config: JoinerConfig) -> None:
        super(Joiner, self).__init__()

        # Out_dim of Both Encoder and predictor shall same dim.
        # Basically joiner is just composed with Linear + activation.
        self._input_dim = config.input_dim
        self._output_dim = config.output_dim
        self._inner_dim = config.inner_dim

        self._enc_proj = nn.Linear(self._input_dim, self._output_dim, bias=True)
        self._pre_proj = nn.Linear(self._input_dim, self._output_dim, bias=True)

        if config.activation == "relu":
            self._activation = nn.ReLU()
        elif config.activation == "tanh":
            self._activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation {config.activation}")

        self._use_out_project = config.use_out_project
        if self._use_out_project:
            self._out_projection = nn.Sequential(
                nn.Linear(self._output_dim, self._inner_dim),
                nn.Linear(self._inner_dim, self._output_dim))
        else:
            self._out_projection = nn.Identity()  # Placeholder

        self._log_softmax = nn.LogSoftmax(dim=-1)

        self._blank_token = 0  # 0 is strictly set for both Ctc and Rnnt.
        self._prune_range = config.prune_range
        self._lm_scale = config.lm_scale
        self._am_scale = config.am_scale

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
                am_pruned, lm_pruned, boundary, ranges, simple_loss
        """

        boundary = torch.zeros((encoder_out.size(0), 4),
                               dtype=torch.int64,
                               device=encoder_out.device)
        boundary[:, 2] = target_lengths
        boundary[:, 3] = encoder_out_lengths
        assert len(target.shape) == 2  # (B, U)
        assert predict_out.shape[-1] >= self._output_dim and predict_out.shape[
            -1] >= self._output_dim, "If pruned rnnt loss applied, output dim of encoder and \
                predictor should be mandatorily larger than num of tokens. Please check your config"

        # Pruned rnnt loss strictly required fp32
        simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
            lm=predict_out.to(dtype=torch.float32),
            am=encoder_out.to(dtype=torch.float32),
            symbols=target,
            termination_symbol=self.blank_token,
            lm_only_scale=self._lm_scale,
            am_only_scale=self._am_scale,
            boundary=boundary,
            reduction="mean",
            return_grad=True,
        )

        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=self.prune_range,
        )

        # am_pruned : [B, T, prune_range, C]
        # Im_pruned : [B, T, prune_range, C]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(am=encoder_out,
                                                  lm=predict_out,
                                                  ranges=ranges)
        return am_pruned, lm_pruned, boundary, ranges, simple_loss

    @torch.jit.unused
    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lengths: torch.Tensor,
        predict_out: torch.Tensor,
        target_lengths: torch.Tensor,
        target: torch.Tensor = torch.empty(0, 0)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        # Project both encoder_out and predictor_out into vocab_size
        encoder_out = self._enc_proj(encoder_out)
        predict_out = self._pre_proj(predict_out)

        if self.prune_range > 0:
            assert target.shape[0] == target_lengths.shape[0]
            # Prune with given prune range, return [B, T, prune_range, C]
            encoder_out, predict_out, boundary, ranges, simple_loss = self._do_rnnt_prune(
                encoder_out=encoder_out,
                encoder_out_lengths=encoder_out_lengths,
                predict_out=predict_out,
                target_lengths=target_lengths,
                target=target)
        else:
            # For API consistency
            boundary = None
            ranges = None
            simple_loss = None

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
        output = self._out_projection(activation_out)

        # Use raw output of joiner for rnnt_loss compute since log_softmax will be
        # done within rnnt_loss.
        return output, boundary, ranges, simple_loss

    @torch.jit.export
    @torch.inference_mode(mode=True)
    def streaming_step(self, encoder_out: torch.Tensor,
                       predictor_out: torch.Tensor):
        # Streaming inference step, accept only 1 frame from encoder and beam_size token from
        # predictor to predict next. strictly follow Auto-aggressively strategy
        # encoder_out: (1, 1, D), predictor_out: (beam_size, 1, D).
        assert encoder_out.shape[0] == 1 and encoder_out.shape[1] == 1
        assert predictor_out.shape[1] == 1

        # Project both with configured out_dim in to vocab_size
        encoder_out = self._enc_proj(encoder_out)
        predictor_out = self._pre_proj(predictor_out)

        encoder_out = encoder_out.unsqueeze(2).contiguous()
        predictor_out = predictor_out.unsqueeze(1).contiguous()

        joint_encodings = encoder_out + predictor_out
        activation_out = self._activation(joint_encodings)
        output = self._out_projection(activation_out)

        output = self._log_softmax(output)  #(1, 1, 1, D)
        output = output.squeeze(1).squeeze(1)  #（1, D)
        return output  # Return next token (1, D)

    @torch.jit.export
    @torch.inference_mode(mode=True)
    def sherpa_onnx_streaming_step(self, encoder_out: torch.Tensor,
                                   predictor_out: torch.Tensor):
        # Wrapped forward for Onnx export
        encoder_out = self._enc_proj(encoder_out)  # (N, proj_enc_out_dim)
        predictor_out = self._pre_proj(predictor_out)  # (N, proj_pre_out_dim)

        joint_encodings = encoder_out + predictor_out
        activation_out = self._activation(joint_encodings)
        output = self._out_projection(activation_out)

        return output  # Return next token (1, D)

    def onnx_export(self, export_path, for_mnn=True, for_sherpa=True):
        """ Interface for onnx export. """
        if for_sherpa:
            self._sherpa_onnx_export(export_path=export_path)
        if for_mnn:
            self._mnn_onnx_export(export_path=export_path)

    def _mnn_onnx_export(self, export_path):
        """ Export Onnx model support deploy with mnn deploy. """
        export_filename = os.path.join(export_path,
                                       "joiner_streaming_step.onnx")
        self.train(False)
        self._restore_forward = self.forward  # Restore forward when export done.
        self.forward = self.streaming_step

        input_dim = self._input_dim
        beam_size = 11

        enc_out = torch.rand(1, 1, input_dim, dtype=torch.float32)
        pred_out = torch.rand(beam_size, 1, input_dim, dtype=torch.float32)

        torch.onnx.export(
            self,
            (enc_out, pred_out),
            export_filename,
            verbose=True,
            opset_version=13,
            input_names=["enc_out", "pred_out"],
            output_names=["logit"],
            dynamic_axes={
                "pred_out": {
                    0: "N"
                },
                "logit": {
                    0: "N"
                },
            },
        )

        self.forward = self._restore_forward  # Restore forward method

    def _sherpa_onnx_export(self, export_path):
        """ Export Onnx model support deploy with sherpa-onnx """
        export_filename = os.path.join(export_path, "joiner_sherpa.onnx")
        self.train(False)
        self._restore_forward = self.forward  # Restore forward when export done.
        self.forward = self.sherpa_onnx_streaming_step
        ts_joiner = torch.jit.script(self)

        input_dim = self._input_dim

        enc_out = torch.rand(11, input_dim, dtype=torch.float32)
        pre_out = torch.rand(11, input_dim, dtype=torch.float32)

        torch.onnx.export(
            ts_joiner,
            (enc_out, pre_out),
            export_filename,
            verbose=False,
            opset_version=13,
            input_names=[
                "encoder_out",
                "decoder_out",
            ],
            output_names=["logit"],
            dynamic_axes={
                "encoder_out": {
                    0: "N"
                },
                "decoder_out": {
                    0: "N"
                },
                "logit": {
                    0: "N"
                },
            },
        )

        meta_data = {
            "joiner_dim": str(input_dim),
        }
        self._add_meta_data(filename=export_filename, meta_data=meta_data)
        self.forward = self._restore_forward  # Restore forward method

    def _add_meta_data(self, filename, meta_data):
        """ Add meta data to an ONNX model. It is changed in-place.
            Args: 
                filename: Filename of the ONNX model to be changed.
                meta_data: Key-value pairs.
        """
        model = onnx.load(filename)
        for key, value in meta_data.items():
            meta = model.metadata_props.add()
            meta.key = key
            meta.value = value
        onnx.save(model, filename)
