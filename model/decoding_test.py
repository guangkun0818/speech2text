# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.09
""" Unittest of decoding """

import glog
import unittest
import torch

from torch.nn.utils.rnn import pad_sequence
from parameterized import parameterized

from dataset.utils import TokenizerSetup
from model.decoding import (reference_decoder, batch_search, CtcGreedyDecoding,
                            RnntGreedyDecoding, RnntBeamDecoding,
                            CifGreedyDecoding)
from model.predictor.predictor import Predictor
from model.joiner.joiner import Joiner, JoinerConfig

# (B, T, D) = (2, 8, 5)
_LOGITS = torch.Tensor([[0.6, 0.0, 0.2, 0.1, 0.1, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0,
                         0.0]]).unsqueeze(0).repeat(2, 1, 1)


class TestCtcGreedySearch(unittest.TestCase):
    """ Unittest of CtcGreedySearch with Tokenizer """

    def setUp(self) -> None:
        super(TestCtcGreedySearch, self).__init__()
        # Set up char tokenizer
        self._char_config = {
            "type": "char",
            "config": {
                "labels": ['a', 'b', 'c']
            }
        }
        self._char_tokenizer = TokenizerSetup(self._char_config)

        # Set up subword tokenizer
        self._subword_config = {
            "type": "subword",
            "config": {
                "spm_model": "sample_data/spm/tokenizer.model",
                "spm_vocab": "sample_data/spm/tokenizer.vocab"
            }
        }
        self._subword_tokenizer = TokenizerSetup(self._subword_config)
        self._char_decode_session = CtcGreedyDecoding(
            tokenizer=self._char_tokenizer)
        self._subword_decode_session = CtcGreedyDecoding(
            tokenizer=self._subword_tokenizer)

    # Params: (ground_truth, labels, tokenzier)
    @parameterized.expand([
        (torch.Tensor([[2, 2, 3, 3], [2, 3, 4, 0]]), ["aabb", "abc"]),
    ])
    def test_decode_reference_tensor_char(self, ground_truth, labels):
        # Unittest of decoding reference tensor, char based
        decoded = reference_decoder(ground_truth, self._char_tokenizer)
        self.assertEqual(decoded, labels)

    # Params: (log_probs, input_length, refers)
    @parameterized.expand([(_LOGITS, torch.Tensor([8,
                                                   8]).long(), ["abc", "abc"]),
                           (_LOGITS, torch.Tensor([8, 3]).long(), ["abc",
                                                                   "a"])])
    def test_greedy_search_char(self, log_probs, input_length, refers):
        # Unittest of ct greedy search, char based
        glog.info(log_probs.shape)
        hypos = batch_search(hidden_states=log_probs,
                             inputs_length=input_length,
                             decode_session=self._char_decode_session)
        glog.info("Ctc decoded: {}".format(hypos))
        self.assertEqual(hypos, refers)

    # Params (refers: List of reference text)
    @parameterized.expand([(["abc", "aabc", "i love china",
                             "bilibili is good"],)])
    def test_decode_reference_tensor_subword(self, refers):
        # Unittest of subword-based decode_reference_tensor
        ref_tensor = []
        for ref in refers:
            vector = self._subword_tokenizer.encode(ref)
            ref_tensor.append(vector)
        batch = pad_sequence(ref_tensor, batch_first=True, padding_value=0)
        decoded = reference_decoder(batch, self._subword_tokenizer)
        self.assertEqual(decoded, refers)

    def test_greedy_search_subword(self):
        # Unittest of ctc greedy search, subword based
        # Setup demo testdata as ["<blank_id>", "<blank_id>", "▁that", "▁i", "▁i", "<blank_id>", "s"]
        # which encoded will be [0, 0, 24, 30, 30, 0, 2]
        log_probs = torch.zeros(1, 7, 128)
        log_probs[0][0][0] = 1  # <blank_id>
        log_probs[0][1][0] = 1  # <blank_id>
        log_probs[0][2][24] = 1  # ▁that
        log_probs[0][3][30] = 1  # ▁i
        log_probs[0][4][30] = 1  # ▁i
        log_probs[0][5][0] = 1  # <blank_id>
        log_probs[0][6][2] = 1  # s
        log_probs = log_probs.repeat(2, 1, 1)  # (2, 6, 128)
        input_length = torch.Tensor([7, 7]).long()
        refers = ["that is", "that is"]

        glog.info(log_probs.shape)
        hypos = batch_search(hidden_states=log_probs,
                             inputs_length=input_length,
                             decode_session=self._subword_decode_session)
        self.assertEqual(hypos, refers)


class TestRnntGreedyDecoding(unittest.TestCase):
    """ Unittest of Rnnt Greedy Decoding """

    def setUp(self) -> None:
        self._tokenzier_config = {
            "type": "subword",
            "config": {
                "spm_model": "sample_data/spm/tokenizer.model",
                "spm_vocab": "sample_data/spm/tokenizer.vocab"
            }
        }
        self._tokenzier = TokenizerSetup(self._tokenzier_config)

        self._predictor_config = {
            "model": "Lstm",
            "config": {
                "num_symbols": 128,
                "output_dim": 512,
                "symbol_embedding_dim": 256,
                "num_lstm_layers": 3,
                "lstm_hidden_dim": 256,
            }
        }
        self._predictor = Predictor(config=self._predictor_config)

        # Input dim should be eq to predictor output_dim, output_dim
        # should be same as num_symbols of predictor.
        self._joiner_config = {
            "input_dim": 512,
            "output_dim": 128,
            "activation": "relu"
        }
        self._joiner = Joiner(config=JoinerConfig(**self._joiner_config))
        self._decode_session = RnntGreedyDecoding(predictor=self._predictor,
                                                  joiner=self._joiner,
                                                  tokenizer=self._tokenzier,
                                                  max_token_step=1)

    def test_rnnt_greedy_decode(self):
        hidden_states = torch.rand(2, 64, 512)
        input_lengths = torch.Tensor([128, 80]).long()
        result = batch_search(hidden_states=hidden_states,
                              inputs_length=input_lengths,
                              decode_session=self._decode_session)
        glog.info(result)


class TestRnntBeamDecoding(unittest.TestCase):
    """ Unittest of rnnt beam decoding """

    def setUp(self) -> None:
        self._tokenzier_config = {
            "type": "subword",
            "config": {
                "spm_model": "sample_data/spm/tokenizer.model",
                "spm_vocab": "sample_data/spm/tokenizer.vocab"
            }
        }
        self._tokenzier = TokenizerSetup(self._tokenzier_config)

        self._predictor_config = {
            "model": "Lstm",
            "config": {
                "num_symbols": 128,
                "output_dim": 512,
                "symbol_embedding_dim": 256,
                "num_lstm_layers": 3,
                "lstm_hidden_dim": 256,
            }
        }
        self._predictor = Predictor(config=self._predictor_config)

        # Input dim should be eq to predictor output_dim, output_dim
        # should be same as num_symbols of predictor.
        self._joiner_config = {
            "input_dim": 512,
            "output_dim": 128,
            "activation": "relu"
        }
        self._joiner = Joiner(config=JoinerConfig(**self._joiner_config))
        self._decode_session = RnntBeamDecoding(predictor=self._predictor,
                                                joiner=self._joiner,
                                                tokenizer=self._tokenzier,
                                                beam_size=4,
                                                cutoff_top_k=4)

    def test_rnnt_beam_decode(self):
        hidden_states = torch.rand(1, 64, 512)
        result = self._decode_session.decode(hidden_states)
        glog.info(result)


class TestCifGreedyDecoding(unittest.TestCase):

    def setUp(self) -> None:
        self._tokenzier_config = {
            "type": "subword",
            "config": {
                "spm_model": "sample_data/spm/tokenizer.model",
                "spm_vocab": "sample_data/spm/tokenizer.vocab"
            }
        }
        self._tokenzier = TokenizerSetup(self._tokenzier_config)
        self._decode_session = CifGreedyDecoding(tokenizer=self._tokenzier)

    def test_cif_greedy_decoding(self):
        hidden_states = torch.rand(4, 100, 128)  # (B, T, label_dim)
        input_lengths = torch.Tensor([23, 80, 100, 64]).long()
        result = batch_search(hidden_states=hidden_states,
                              inputs_length=input_lengths,
                              decode_session=self._decode_session)
        glog.info(result)


if __name__ == "__main__":
    unittest.main()
