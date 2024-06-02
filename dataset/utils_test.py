# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.01
""" Unittest of Dataset Utilities """

import unittest
import glog
import torch

from parameterized import parameterized

from dataset.utils import CharTokenizer, CharTokenizerConfig
from dataset.utils import SubwordTokenizer, SubwordTokenizerConfig


class TestCharTokenizer(unittest.TestCase):
    """ Unittest of Char Tokenzier """

    def setUp(self) -> None:

        self._config = {
            "labels": [
                "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
                "'", " "
            ]
        }

        self._tokenizer = CharTokenizer(config=CharTokenizerConfig(
            **self._config))

    def test_num_labels(self):
        self.assertEqual(
            len(self._config["labels"]) + 3, len(self._tokenizer.labels))
        self.assertIn("<blank_id>", self._tokenizer.labels)
        self.assertEqual("<blank_id>", self._tokenizer.labels[0])
        self.assertIn("<sos/eos>", self._tokenizer.labels)
        self.assertEqual("<sos/eos>", self._tokenizer.labels[-1])

    @parameterized.expand([
        ("season with salt and pepper and a little sugar to taste"),
        ("struggle warfare was the condition of private ownership it was fatal"
        ),
        ("i don't believe ann knew any magic or she'd have worked it before"),
        ("a black drove came up over the hill behind the wedding party")
    ])
    def test_encode_as_tensor(self, text):
        # Unittest of encode and decode
        glog.info(text)
        labels = self._tokenizer.encode(text)
        decoded = self._tokenizer.decode(labels)

        self.assertEqual(text, decoded)

    @parameterized.expand([
        ("season with salt and pepper and a little sugar to taste"),
        ("struggle warfare was the condition of private ownership it was fatal"
        ),
        ("i don't believe ann knew any magic or she'd have worked it before"),
        ("a black drove came up over the hill behind the wedding party")
    ])
    def test_encoder_as_tokens(self, text):
        # Unittest of encode_as_tokens and decode_from_tokens
        glog.info(text)
        tokens = self._tokenizer.encode_as_tokens(text)
        decoded = self._tokenizer.decode_from_tokens(tokens)

        self.assertEqual(text, decoded)


class TestSubwordTokenizer(unittest.TestCase):
    """ Unittest of Subword Tokenizer """

    def setUp(self) -> None:
        self._config = {
            "spm_model": "sample_data/spm/tokenizer.model",
            "spm_vocab": "sample_data/spm/tokenizer.vocab"
        }

        self._tokenizer = SubwordTokenizer(config=SubwordTokenizerConfig(
            **self._config))

    def test_num_labels(self):
        self.assertNotIn("<s>", self._tokenizer.labels)
        self.assertNotIn("</s>", self._tokenizer.labels)
        self.assertIn("<blank_id>", self._tokenizer.labels)
        self.assertEqual("<blank_id>", self._tokenizer.labels[0])
        self.assertIn("<sos/eos>", self._tokenizer.labels)
        self.assertEqual("<sos/eos>", self._tokenizer.labels[-1])

    @parameterized.expand([
        ("season with salt and pepper and a little sugar to taste"),
        ("struggle warfare was the condition of private ownership it was fatal"
        ),
        ("i don't believe ann knew any magic or she'd have worked it before"),
        ("a black drove came up over the hill behind the wedding party")
    ])
    def test_encode_as_tensor(self, text):
        # Unittest of encode and decode
        glog.info(text)
        labels = self._tokenizer.encode(text)
        decoded = self._tokenizer.decode(labels)

        self.assertEqual(text, decoded)

    @parameterized.expand([
        ("season with salt and pepper and a little sugar to taste"),
        ("struggle warfare was the condition of private ownership it was fatal"
        ),
        ("i don't believe ann knew any magic or she'd have worked it before"),
        ("a black drove came up over the hill behind the wedding party")
    ])
    def test_encoder_as_tokens(self, text):
        # Unittest of encode_as_tokens and decode_from_tokens
        glog.info(text)
        tokens = self._tokenizer.encode_as_tokens(text)
        decoded = self._tokenizer.decode_from_tokens(tokens)

        self.assertEqual(text, decoded)


if __name__ == "__main__":
    unittest.main()
