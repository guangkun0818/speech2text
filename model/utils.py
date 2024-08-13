# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.09
""" Metrics for model evaluation. """

import dataclasses
import glog
import random
import torch

from typing import List

from dataset.utils import Tokenizer
from model.predictor.predictor import Predictor
from model.joiner.joiner import Joiner

from model.decoding import (reference_decoder, batch_search, RnntGreedyDecoding,
                            CtcGreedyDecoding)

from typing import Tuple


def _levenshtein(a: List, b: List) -> int:
    """ Calculates the Levenshtein distance between a and b.
        The code was copied from: http://hetland.org/coding/python/levenshtein.py
    """
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


def word_error_rate(hypotheses: List[str],
                    references: List[str],
                    show_on_screen=True,
                    use_cer=False) -> float:
    """ Computes Average Word Error Rate between two texts represented as 
        corresponding lists of string. Hypotheses and references must have 
        same length.
        Args:
            hypotheses: list of hypotheses 
            references: list of references
            show_on_screen: Indicating show result on screen, works when 
                            compute total wer avoiding repeatedly export to log file
            use_cer: bool, set True to enable cer
        Returns:
            WER: float, average word error rate
    """
    scores = 0
    words = 0
    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and references"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses),
                                                 len(references)))
    # Random pick one from eval batch and show on screen
    if show_on_screen:
        index_picked = random.randint(0, len(references) - 1)
        glog.info("Pre: {}".format(hypotheses[index_picked]))
        glog.info("Ref: {}".format(references[index_picked]))

    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()
        words += len(r_list)
        scores += _levenshtein(h_list, r_list)
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float('inf')
    return wer


@dataclasses.dataclass
class AsrMetricConfig:
    """ Metric Config supporting Ctc and Rnnt task. """
    decode_method: str = "ctc_greedy_search"
    max_token_step: int = 5  # Max Token Step of Rnnt Decoder


class AsrMetric(object):
    """ Metric for training """

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: AsrMetricConfig,
        predictor: Predictor = None,
        joiner: Joiner = None,
    ):

        self._tokenizer = tokenizer
        # TODO: future will support BeamSearch
        if config.decode_method == "ctc_greedy_search":
            self._decode_sess = CtcGreedyDecoding(tokenizer=self._tokenizer)
        elif config.decode_method == "rnnt_greedy_search":
            self._decode_sess = RnntGreedyDecoding(
                tokenizer=self._tokenizer,
                predictor=predictor,
                joiner=joiner,
                max_token_step=config.max_token_step)
        else:
            NotImplementedError

    def __call__(self, hidden_states, inputs_length, ground_truth):
        """ Call func of Metric
            Args:
                hidden_states: Tensor(Batch, Seq_len, Hidden_dim) 
                inputs_length: Tensor(Batch)
                ground_truth: Ground_truth of labels, tokenized.
        """
        references = reference_decoder(ground_truth, self._tokenizer)
        hypotheses = batch_search(hidden_states, inputs_length,
                                  self._decode_sess)
        wer = word_error_rate(hypotheses=hypotheses, references=references)
        return wer


@dataclasses.dataclass
class SslMetricConfig:
    """ Config of SSl Metrics, Top_k acc.
    """
    top_ks: Tuple[int] = (1, 5)


class SslMetric(object):
    """ Metric for ssl training """

    def __init__(self, config: SslMetricConfig):
        # Initialization
        self._compute_acc = self._ssl_accuarcy
        self._top_ks = config.top_ks

    def _ssl_accuarcy(self, logits: torch.Tensor, labels: torch.Tensor,
                      masked_dim: torch.Tensor, top_k):
        """ Compute ACC with given top_k of SSL task 
            Args:
                logits: torch.Size(B, T, D)
                labels: torch.Size(B, T)
                masked_dim: torch.Size(B, T)
                top_k: Top_k ACC
        """
        batch_size = labels.shape[0]

        logits_top_k = logits.topk(
            top_k, dim=-1, largest=True,
            sorted=True).indices  # logits_top_k shape: (B, T, top_k)

        # Masked_dim, where 1 indicate position to be predicted, therefore, mask out 0 position
        # for metrics (ACC) evaluation.
        valid_logits_top_k = logits_top_k.masked_fill(
            (1 - masked_dim).unsqueeze(-1).bool(), value=-1)

        valid_labels = masked_dim * labels
        valid_labels = valid_labels.unsqueeze(-1)

        # Compute num matches within batch
        num_matched = torch.eq(valid_logits_top_k, valid_labels).sum().float()

        # ACC of top_k
        return num_matched / (masked_dim.sum() + 1e-7
                             )  # In case inf acc due to no mask within batch

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor,
                 masked_dim: torch.Tensor):
        """ Call func of Metric """
        # Return metrics as dic during eval stage
        metrics = {}
        for k in self._top_ks:
            metrics["top_{}_acc".format(k)] = self._compute_acc(
                logits=logits, labels=labels, masked_dim=masked_dim, top_k=k)
        return metrics
