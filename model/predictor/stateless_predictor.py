# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.07.28
""" Stateless Predictor impl, based on
    https://arxiv.org/pdf/2109.07513
"""

import os
import dataclasses
import torch
import onnx
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Tuple


@dataclasses.dataclass
class StatelessPredictorConfig:
    """ Stateless Predictor Config """
    num_symbols: int = 128  # ‹blank_id> = 0, <sos> = num_symbols - 1 are taken into account
    output_dim: int = 1024
    symbol_embedding_dim: int = 512
    context_size: int = 5


class StatelessPredictor(nn.Module):
    """ Stateless Predictor, replace LSTM based predictor into Embedding + Conv1d """

    def __init__(self, config: StatelessPredictorConfig) -> None:
        super(StatelessPredictor, self).__init__()

        # NOTE: Start of sentence token <sos>. Last token of token list
        self._sos_token = config.num_symbols - 1
        self._blank_token = 0  # 0 is strictly set for both Ctc and Rnnt.

        self._embedding_dim = config.symbol_embedding_dim
        self._num_symbols = config.num_symbols
        self._embedding = nn.Embedding(num_embeddings=self._num_symbols,
                                       embedding_dim=self._embedding_dim)
        assert config.context_size >= 1, "context_size should be greater than or eq to 1"
        self._context_size = config.context_size
        self._output_dim = config.output_dim

        # Context layer using Conv1d with kernel_size = context_size to impl n-gram-like lm.
        self._conv = nn.Conv1d(
            in_channels=self._embedding_dim,
            out_channels=self._embedding_dim,
            kernel_size=self._context_size,
            stride=1,
            padding=0,
            groups=self._embedding_dim,
            bias=False,
        )
        self._output_linear = nn.Linear(self._embedding_dim, self._output_dim)

    @property
    def sos_token(self) -> int:
        return self._sos_token

    @property
    def blank_token(self) -> int:
        return self._blank_token

    def _left_padding(self, x: torch.Tensor) -> torch.Tensor:
        # left padding input tokens with <blank_id>, only when training.
        # TODO: Future can use <sos> and <eos> to modeling the completeness
        # of Sentence to get preciser endpoint where left_padding <sos> and
        # right padding <eos>
        assert len(x.shape) == 2  # (B, U)
        return F.pad(x.float(), (1, 0, 0, 0), value=float(self.blank_token)).to(
            torch.int32)  # (B, 1 + U)

    def forward(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Training Graph, works only for training
            Args:
                input: Tokens inputs, (B, U) 
                length: Tokens lengths, (B)
                state: Init state, this only works during training. (1, context - 1)
        """
        bs = input.shape[0]
        state = state.repeat(bs, 1).to(input.device)  # (B, context_size - 1)
        input = self._left_padding(input)
        ctxed_input = torch.concat([state, input],
                                   dim=1)  # (B, context_size - 1 + 1 + U)
        cache_pos = ctxed_input.shape[1] - self._context_size
        out_state = ctxed_input[:, cache_pos:]

        embs = self._embedding(ctxed_input)
        embs = embs.contiguous().transpose(1, 2)  # (B, U, D) -> (B, D, U)
        output = self._conv(embs).contiguous().transpose(1, 2)
        output = self._output_linear(output)

        return output, lengths, out_state

    @torch.jit.export
    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        # Initialize state when As Session start, init state is
        # [blank_token, ..., blank_token] with length = context_size - 1
        return torch.zeros(batch_size, self._context_size - 1).to(torch.int32)

    @torch.jit.export
    @torch.inference_mode(mode=True)
    def streaming_step(
            self, input: torch.Tensor,
            state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Streaming inference step of Predictor, accept 1 token input only,
        # (Batch_size, 1, D), auto-aggressive, you gotcha.
        assert input.shape[1] == 1  # only support sequence length = 1
        ctxed_input = torch.concat([state.to(input.device), input],
                                   dim=1)  # (1, context_size - 1 + 1 + U)
        cache_pos = ctxed_input.shape[1] - self._context_size + 1
        out_state = ctxed_input[:, cache_pos:]

        embs = self._embedding(ctxed_input)
        embs = embs.contiguous().transpose(1, 2)  # (B, U, D) -> (B, D, U)

        output = self._conv(embs).contiguous().transpose(1, 2)
        output = self._output_linear(output)
        return output, out_state  # Output

    @torch.jit.export
    @torch.inference_mode(mode=True)
    def sherpa_onnx_streaming_step(self, input: torch.Tensor):
        # Wrapped forward for Onnx export
        assert input.shape[1] == self._context_size

        embs = self._embedding(input)
        embs = embs.contiguous().transpose(1, 2)  # (B, U, D) →> (B, D, U)
        output = self._conv(embs).contiguous().transpose(1, 2)
        output = self._output_linear(output).squeeze(1)
        return output  # Output (B, D)

    def onnx_export(self, export_path, for_mnn=True, for_sherpa=True):
        """ Interface for onnx export. """
        if for_sherpa:
            self._sherpa_onnx_export(export_path=export_path)
        if for_mnn:
            self._mnn_onnx_export(export_path=export_path)

    def _mnn_onnx_export(self, export_path):
        """ Export Onnx model support deploy with mnn deploy, predictor seperated into init model 
            and streaming_step model respectivly.
        """

        init_model_filename = os.path.join(export_path, "predictor_init.onnx")
        self.train(False)
        self._restore_forward = self.forward  # Restore forward when export done.

        # Predictor init model.
        self.forward = self.init_state
        # Args is required by onnx export, set as empty for init
        torch.onnx.export(self,
                          args=(),
                          f=init_model_filename,
                          verbose=True,
                          opset_version=13,
                          input_names=None,
                          output_names=["states"])

        # Predictor streaming_step model.
        self.forward = self.streaming_step
        streaming_step_model_filename = os.path.join(
            export_path, "predictor_streaming_step.onnx")
        batch_size = 10
        prev_states = self.init_state(batch_size)
        pred_in = torch.randint(1, 128, (batch_size, 1))  # (B, 1)
        torch.onnx.export(self,
                          args=(pred_in, prev_states),
                          f=streaming_step_model_filename,
                          verbose=True,
                          opset_version=13,
                          input_names=["pred_in", "prev_states"],
                          output_names=["pred_out", "next_states"])

        self.forward = self._restore_forward  # Restore forward method

    def _sherpa_onnx_export(self, export_path):
        """ Export Onnx model support deploy with sherpa-onnx """
        export_filename = os.path.join(export_path, "predictor.onnx")
        self.train(False)
        self._restore_forward = self.forward  # Restore forward when export done.
        self.forward = self.sherpa_onnx_streaming_step
        ctx_size = self._context_size
        vocab_size = self._num_symbols

        y = torch.zeros(10, ctx_size, dtype=torch.int64)
        torch.onnx.export(
            self,
            y,
            export_filename,
            verbose=False,
            opset_version=13,
            input_names=["y"],
            output_names=["decoder_out"],
            dynamic_axes={
                "y": {
                    0: "N"
                },
                "decoder_out": {
                    0: "N"
                },
            },
        )

        meta_data = {
            "context_size": str(ctx_size),
            "vocab_size": str(vocab_size),
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
