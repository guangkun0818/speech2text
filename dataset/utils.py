# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.03.29
""" Necessary utilities of Dataset, including Tokenizer 
    and collate_fn impl. More to dev. 
"""

import abc
import dataclasses
import torch
import sentencepiece as spm

from typing import List, Dict
from torch.nn.utils.rnn import pad_sequence


class Tokenizer(abc.ABC):
    """ Abstract class template for different tokenizer support """

    @property
    @abc.abstractmethod
    def labels(self) -> List[str]:
        """ return all tokens """
        ...

    @abc.abstractmethod
    def encode(self, text: str) -> torch.Tensor:
        """ Encode raw text as vector """
        ...

    @abc.abstractmethod
    def decode(self, vector: torch.Tensor) -> str:
        """ Decode list of tokens as raw text """
        ...

    @abc.abstractmethod
    def encode_as_tokens(self, text: str) -> List[str]:
        # Encode text as tokens rather than vector, used during inference
        ...

    @abc.abstractmethod
    def decode_from_tokens(self, tokens: List[str]) -> str:
        # Decode list of tokens as raw text, used during inference
        ...

    def export_units(self, export_filename: str) -> None:
        with open(export_filename, 'w') as units_f:
            for i, unit in enumerate(self.labels):
                units_f.write("{} {}\n".format(unit, i))

    def _text_to_vector(self, text: List[str]) -> torch.Tensor:
        """ Return: torch.Tensor(nums_tokens) """
        vector = []
        for char in text:
            if char not in self.labels:
                vector.append(self.labels.index("<unk>"))
            else:
                vector.append(self.labels.index(char))
        vector = torch.Tensor(vector).long()
        return vector

    def _vector_to_tokens(self, vector: torch.Tensor) -> List[str]:
        """ Return: tokens (seq_lens of vector) """
        tokens = []
        for idx in vector.tolist():
            tokens.append(self.labels[idx])
        return tokens


@dataclasses.dataclass
class CharTokenizerConfig:
    """ Char based Tokenizer Config, should be configured from yaml """
    labels: tuple = ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
                     "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x",
                     "y", "z", "'", " ")


@dataclasses.dataclass
class SubwordTokenizerConfig:
    """ Subword based Tokenzier Config, spm_model and spm_vocab should 
        provided in yaml. 
    """
    spm_model: str = None
    spm_vocab: str = None


class SubwordTokenizer(Tokenizer):
    """ Subword based tokenzier impl """

    def __init__(self, config: SubwordTokenizerConfig):
        super(SubwordTokenizer, self).__init__()

        assert config.spm_model is not None
        assert config.spm_vocab is not None
        self._spm_model = config.spm_model
        self._spm_vocab = config.spm_vocab
        self._labels = self._load_labels(self._spm_vocab)
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(self._spm_model)

    @property
    def labels(self) -> List[str]:
        return self._labels

    def _load_labels(self, vocabs) -> List[str]:
        labels = ["<blank_id>"]
        with open(vocabs, 'r') as vocabs_f:
            for line in vocabs_f:
                token = line.strip().split("\t")[0]
                # Get rid of <s> and </s>
                if token != "<s>" and token != "</s>":
                    labels.append(token)
        labels.append("<sos/eos>")  # add <sos/eos> token for rescore and Rnnt
        return labels

    def encode(self, text: str) -> torch.Tensor:
        return self._text_to_vector(
            self._sp.EncodeAsPieces(text, emit_unk_piece=True))

    def decode(self, vector: torch.Tensor) -> str:
        return self._sp.DecodePieces(self._vector_to_tokens(vector))

    def encode_as_tokens(self, text: str) -> List[str]:
        tokens = self._sp.EncodeAsPieces(text, emit_unk_piece=True)
        for i in range(len(tokens)):
            if tokens[i] not in self.labels:
                tokens[i] = "<unk>"
        return tokens

    def decode_from_tokens(self, tokens: List[str]) -> str:
        for char in tokens:
            assert char in self.labels, "Out of vocabulary detects with '{}'".format(
                char)
        return self._sp.DecodePieces(tokens)


class CharTokenizer(Tokenizer):
    """ Char based tokenzier impl """

    def __init__(self, config: CharTokenizerConfig) -> None:
        super(CharTokenizer, self).__init__()

        # Insert <blank_id> at front and <sos> at rear
        self._labels = ["<blank_id>", "<unk>"] + config.labels + ["<sos/eos>"]

    @property
    def labels(self):
        return self._labels

    def encode(self, text: str) -> torch.Tensor:
        return self._text_to_vector(text)

    def decode(self, vector: torch.Tensor) -> str:
        return ''.join(self._vector_to_tokens(vector))

    def encode_as_tokens(self, text: str) -> List[str]:
        tokens = list(text)
        for i in range(len(tokens)):
            if tokens[i] not in self.labels:
                tokens[i] = "<unk>"
        return tokens

    def decode_from_tokens(self, tokens: List[str]) -> str:
        for char in tokens:
            assert char in self.labels, "Out of vocabulary detects with '{}'".format(
                char)
        return ''.join(tokens)


def TokenizerSetup(config) -> Tokenizer:
    """ Interface of Tokenizer Setup """
    if config["type"] == "char":
        return CharTokenizer(config=CharTokenizerConfig(**config["config"]))
    elif config["type"] == "subword":
        return SubwordTokenizer(config=SubwordTokenizerConfig(
            **config["config"]))
    else:
        raise ValueError(
            "Only 'char' and 'subword' tokenizer supported currently.")


def batch(batch: Dict[str, List]) -> Dict[str, torch.Tensor]:
    """ Padding dynamic length of inputs and stack as off-to-go batch. """
    if "feat" in batch and "label" in batch:
        batch["feat"] = pad_sequence(batch["feat"],
                                     batch_first=True,
                                     padding_value=0)
        batch["feat_length"] = torch.Tensor(batch["feat_length"]).long()
        batch["label"] = pad_sequence(batch["label"],
                                      batch_first=True,
                                      padding_value=0)
        batch["label_length"] = torch.Tensor(batch["label_length"]).long()
    elif "raw_feat" in batch and "auged_feat" in batch:
        batch["raw_feat"] = pad_sequence(batch["raw_feat"],
                                         batch_first=True,
                                         padding_value=0)
        batch["auged_feat"] = pad_sequence(batch["auged_feat"],
                                           batch_first=True,
                                           padding_value=0)
        batch["feat_length"] = torch.Tensor(batch["feat_length"]).long()

    return batch
