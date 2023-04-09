# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.09
""" Decoding factory for eval or inference, Ctc 
    and Rnnt decoder included.
"""

import torch

from typing import List
from dataset.utils import Tokenizer
from model.predictor.predictor import Predictor
from model.joiner.joiner import Joiner


def _ctc_decoder_predictions_tensor(tensor: torch.Tensor, tokenizer: Tokenizer):
    """ Decodes a sequence of labels to words """
    # Default blank_id is 0 in ctc loss
    blank_id = 0
    hypotheses = []
    prediction_cpu_tensor = tensor.long().cpu()
    # iterate over batch
    for batch_id in range(prediction_cpu_tensor.shape[0]):
        prediction = prediction_cpu_tensor[batch_id].numpy().tolist()
        # CTC decoding procedure
        decoded_prediction = []
        previous = 0  # id of a blank symbol
        for p in prediction:
            if (p != previous or previous == blank_id) and p != blank_id:
                decoded_prediction.append(p)
            previous = p
        # Tokenizer will decode Tensor of token_id into text
        hypothesis = tokenizer.decode(torch.Tensor(decoded_prediction).long())
        hypotheses.append(hypothesis)
    return hypotheses


def reference_decoder(tensor: torch.Tensor, tokenizer: Tokenizer):
    """ Specifically decoder one-hotified labels as text for
        wer compute during eval stage
    """
    # Might be changed in the future
    padding_id = 0
    references = []
    reference_cpu_tensor = tensor.long().cpu()
    # iterate over batch
    for batch_id in range(reference_cpu_tensor.shape[0]):
        reference_lst = reference_cpu_tensor[batch_id].numpy().tolist()
        decoded_ref = []
        for unit in reference_lst:
            # When padding id encountered, should stop
            if unit == padding_id:
                break
            decoded_ref.append(unit)
        # Tokenizer will decode Tensor of token_id into text
        ref = tokenizer.decode(torch.Tensor(decoded_ref).long())
        references.append(ref)
    return references


def ctc_greedy_search(log_probs: torch.Tensor, inputs_length: torch.Tensor,
                      tokenizer: Tokenizer):
    """ Ctc Greedy search of log_probs
        Args:
            log_probs: (batch, seq_len, label_dim) 
            inputs_length (batch)
            tokenizer: training tokenizer (char or subward, or more)
        return:
            hypotheses: List[str], List of decoded text of whole batches.
    """
    assert log_probs.shape[-1] == len(tokenizer.labels)
    greedy_result = torch.argmax(log_probs, dim=2, keepdim=True).squeeze(-1)

    # Generate mask of greedy search result for padding applied in batch
    mask = torch.arange(inputs_length.max()).unsqueeze(0).to(
        inputs_length.device) > inputs_length.unsqueeze(-1) - 1
    # Fill the padding parts as 0, default blank_id
    greedy_result = greedy_result.masked_fill(mask, 0).long()
    hypotheses = _ctc_decoder_predictions_tensor(greedy_result, tokenizer)
    return hypotheses


class RnntGreedyDecoding(object):
    """ Internal Rnnt Greedy decoding
        Args:
            tokenizer: training tokenizer (char or subword, or more) 
            predictor: Predictor Module of Rnnt 
            joiner: Joiner Module of Rnnt
    """

    def __init__(self,
                 tokenizer: Tokenizer,
                 predictor: Predictor,
                 joiner: Joiner,
                 max_token_step=10):

        self._tokenizer = tokenizer
        self._predictor = predictor
        self._joiner = joiner
        # Limit token step (lattice move upward) within one time step. This because
        # decoding will cost too much time on eval stage when model just inited.
        self._max_token_step = max_token_step
        assert hasattr(self._predictor, "streaming_step") and hasattr(
            self._joiner, "streaming_step"
        ), "Predictor and Joiner should impl streaming_step for decoding."

    def _check_blank(self, pred_token: torch.Tensor) -> bool:
        # Internal func to check if predicted token is <blank_id>, whose index == 0
        return torch.allclose(pred_token,
                              torch.zeros(1, 1).long().to(pred_token.device))

    def decode(self, hidden_states: torch.Tensor) -> str:
        # Rnnt greedy decoding. Here only support Batchsize = 1
        # hidden_states: (1, T, D), encoder output

        assert hidden_states.shape[0] == 1, "Support BatchSize = 1 only."

        # Decoding setup, init predictor state, joiner is cache-free. Deocding
        # process should be started with <blank_id>, <sos> maybe in future.
        pred_state = self._predictor.init_state()
        tot_time_steps = hidden_states.shape[1]
        curr_time_step = 0
        curr_token = torch.Tensor([[0]]).long().to(
            hidden_states.device)  # init_token is <blank_id> (1, 1)
        num_token_step = 0  # Num of token step with in one time step
        # Compute pred out in advance
        pred_out, pred_state = self._predictor.streaming_step(
            curr_token, pred_state)
        decoded_result = []  # cache result.

        while curr_time_step < tot_time_steps:
            # After consuming all hidden_state, decoding end.
            enc_out = hidden_states[:, curr_time_step:curr_time_step +
                                    1, :]  # (1, 1, D)
            pred_token = self._joiner.streaming_step(enc_out,
                                                     pred_out)  # (1, D)
            pred_token = pred_token.argmax(dim=-1).unsqueeze(
                0)  # pick the argmax greedily
            if self._check_blank(
                    pred_token) or num_token_step > self._max_token_step:
                # If <blank_id> predicted or num token step reach the predefined limit of
                # max_token_step, move to next time_step (lattice move rightward). Token_step
                # maintained.
                curr_time_step += 1
                num_token_step = 0
                continue
            else:
                # If not <blank_id>, move to next token_step (lattice move upward)
                # time_step maintained, num_token_step update.
                num_token_step += 1
                curr_token = pred_token
                pred_out, pred_state = self._predictor.streaming_step(
                    curr_token, pred_state)
                decoded_result.append(pred_token.item())
                continue

        # Return decoded text
        return self._tokenizer.decode(torch.Tensor(decoded_result).long())


def rnnt_greedy_search(hidden_states: torch.Tensor, inputs_length: torch.Tensor,
                       decode_session: RnntGreedyDecoding):
    """ Rnnt Greedy Search.
        Args:
            hidden_states: (batch, seq_len, encoder_out_dim) 
            inputs_length (batch)
            decode_session: Built instance of RnntGreedyDecoding
        return:
            hypotheses: List[str], List of decoded text of whole batches.
    """
    results = []
    for entry_id in range(hidden_states.shape[0]):
        actual_1 = inputs_length[entry_id].item()
        actual_hidden_s = hidden_states[entry_id:entry_id + 1, :actual_1, :]
        results.append(decode_session.decode(actual_hidden_s))

    return results
