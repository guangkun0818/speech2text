# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.04.09
""" Decoding factory for eval or inference, Ctc 
    and Rnnt decoder included.
"""

import glog
import os
import abc
import dataclasses
import math
import torch

from enum import Enum, unique
from typing import List, Optional
from torchaudio.models.decoder import ctc_decoder

from dataset.utils import Tokenizer
from model.predictor.predictor import Predictor
from model.joiner.joiner import Joiner


class DecodingMethod(abc.ABC):
    """ Abstract class for decoding method """

    @abc.abstractmethod
    def decode(self, hidden_states: torch.Tensor) -> str:
        pass


def batch_search(hidden_states: torch.Tensor, inputs_length: torch.Tensor,
                 decode_session: DecodingMethod):
    """ Factory method for batch decoding.
        Args:
            hidden_states: (batch, seq_len, encoder_out_dim) 
            inputs_length (batch)
            decode_session: Built instance of DecodingMethod
        return:
            hypotheses: List[str], List of decoded text of whole batches.
    """
    results = []
    for entry_id in range(hidden_states.shape[0]):
        actual_1 = inputs_length[entry_id].item()
        actual_hidden_s = hidden_states[entry_id:entry_id + 1, :actual_1, :]
        results.append(decode_session.decode(actual_hidden_s))

    return results


class CtcGreedyDecoding(DecodingMethod):
    """ CTC greedy decoding impl """

    def __init__(self, tokenizer: Tokenizer) -> None:
        super(CtcGreedyDecoding, self).__init__()

        self._tokenizer = tokenizer

    def decode(self, hidden_states: torch.Tensor) -> str:
        """ Decodes a sequence of labels to words """
        assert hidden_states.shape[0] == 1, "Support BatchSize = 1 only."
        assert hidden_states.shape[-1] == len(self._tokenizer.labels)

        # (1, seq_len, label_dim) -> (1, seq_len)
        hidden_states = torch.argmax(hidden_states, dim=2,
                                     keepdim=True).squeeze(-1)

        # Default blank_id is 0 in ctc loss
        blank_id = 0
        prediction_cpu_tensor = hidden_states.long().cpu()
        prediction = prediction_cpu_tensor[0].numpy().tolist()
        # CTC decoding procedure
        decoded_prediction = []
        previous = 0  # id of a blank symbol
        for p in prediction:
            if (p != previous or previous == blank_id) and p != blank_id:
                decoded_prediction.append(p)
            previous = p
        # Tokenizer will decode Tensor of token_id into text
        hypothesis = self._tokenizer.decode(
            torch.Tensor(decoded_prediction).long())
        return hypothesis


class CtcLexiconBeamDecoding(DecodingMethod):
    """ Wrapped ctc_decoder provided from torchaudio, which is build 
        from flashlight-text decoder, https://github.com/flashlight/text. 
    """

    def __init__(self,
                 tokenizer: Tokenizer,
                 nbest: int = 1,
                 beam_size: int = 50,
                 beam_size_token: Optional[int] = None,
                 beam_threshold: float = 50,
                 blank_token: str = "<blank_id>",
                 sil_token: str = "<blank_id>",
                 language_model: str = None,
                 word_list: str = None,
                 export_path: str = None) -> None:

        self._tokenizer = tokenizer
        self._lm_path = language_model
        self._wl_path = word_list
        self._sil_token = sil_token
        self._export_path = export_path

        self._lexicon = None
        if self._lm_path is not None and self._wl_path is not None:
            # If both language model and word list is not none, specify n-gram lm applied.
            self._lexicon = self._generate_lexicon(self._wl_path,
                                                   self._sil_token)

        self._ctc_decoder = ctc_decoder(
            lexicon=self._lexicon,
            tokens=self._tokenizer.labels,
            lm=self._lm_path,
            nbest=nbest,
            beam_size=beam_size,
            beam_size_token=beam_size_token,
            beam_threshold=beam_threshold,
            blank_token=blank_token,
            sil_token=sil_token,
        )

    def _generate_lexicon(self, word_list, sil_token) -> str:
        # Generate Lexicon file if wordlist is specified in config. This
        # will lead to use LexiconDecoder.
        lm_rsrc = os.path.join(self._export_path, "lm_rsrc")
        os.makedirs(lm_rsrc, exist_ok=True)
        lexicon_path = os.path.join(lm_rsrc, "lexicon")

        glog.info("Generating Lexicon from word list....")
        with open(lexicon_path, 'w') as lexicon_f, open(word_list, 'r') as wl_f:
            for line in wl_f:
                word = line.strip()
                assert sil_token == "<blank_id>"
                dict_entry = [word] + self._tokenizer.encode_as_tokens(word)
                lexicon_f.write("{}\n".format(dict_entry))
        glog.info("Generated Lexicon stored in {}".format(lexicon_path))

        return lexicon_path  # Return generate lexicon path

    def _finalize(self, decoded_text):
        # Reverse decoded text by remove delimiter of sil_token
        result = "".join(decoded_text).replace(self._sil_token, " ").strip()
        return result

    def decode(self, hidden_states: torch.Tensor) -> str:
        assert hidden_states.shape[0] == 1, "Support BatchSize = 1 only."
        predicts = self._ctc_decoder(hidden_states.to(torch.float32).cpu())
        result = self._finalize(self._tokenizer.decode(
            predicts[0][0].tokens))  # Decode top 1 within beam.
        return result


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


class CifGreedyDecoding(DecodingMethod):
    """ CIF greedy decoding, Non-autoregressive argmax. """

    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__()
        self._tokenizer = tokenizer

    def decode(self, hidden_states: torch.Tensor) -> str:
        # hidden_states: (1, T, D), decoder output after cif output
        assert hidden_states.shape[0] == 1, "Support BatchSize = 1 only."
        assert hidden_states.shape[-1] == len(self._tokenizer.labels)

        greedy_result = torch.argmax(hidden_states, dim=-1).squeeze(0)
        return self._tokenizer.decode(torch.Tensor(greedy_result))


class RnntGreedyDecoding(DecodingMethod):
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


def _log_add(a, b):
    return math.log(math.exp(a) + math.exp(b))


@dataclasses.dataclass
class DecodedBeam:
    """ Refered as Beam within DecodingState """
    decoded_tokens: List[int] = dataclasses.field(default_factory=list)
    end_with_blank: bool = True  # <blank_id>
    score: float = 0.0
    pred_state: torch.Tensor = None
    pred_out: torch.Tensor = None


@dataclasses.dataclass
class DecodingState:
    """ struct DecodingState of RnntBeamDecoding storing for each beam """
    beams: List[DecodedBeam] = dataclasses.field(default_factory=list)
    best_beam: DecodedBeam = None


class RnntBeamDecoding(DecodingMethod):
    """ Beam search of Rnnt Asr system, restrict token step as 1 taking advantage 
        of peaky behavior of Rnnt loss, which interestingly pretty much similar 
        to CTC.
        
        Args:
            tokenizer: training tokenizer (char or subword, or more) 
            predictor: Predictor Module of Rnnt 
            joiner: Joiner Module of Rnnt 
            beam_size: Beam size
    """

    def __init__(self,
                 tokenizer: Tokenizer,
                 predictor: Predictor,
                 joiner: Joiner,
                 beam_size=4,
                 cutoff_top_k=4) -> None:

        self._tokenizer = tokenizer
        self._predictor = predictor
        self._joiner = joiner

        self._beam_size = beam_size
        self._cutoff_top_k = cutoff_top_k

        self._decoding_state = None

        assert hasattr(self._predictor, "streaming_step") and hasattr(
            self._joiner, "streaming_step"
        ), "Predictor and Joiner should impl streaming_step for decoding."

    def reset(self, device: torch.device):
        # Init decoding state
        init_beam = DecodedBeam()

        # Decoding setup, init predictor state, joiner is cache-free. Deocding
        # process should be start with <blank_id> = 0.
        pred_state = self._predictor.init_state()
        blk_token = torch.Tensor([[0]]).long().to(device)
        pred_out, pred_state = self._predictor.streaming_step(
            blk_token, pred_state)

        init_beam.pred_out = pred_out
        init_beam.pred_state = pred_state

        self._decoding_state = DecodingState(beams=[init_beam],
                                             best_beam=init_beam)

    def _build_beam_pred_out(self) -> torch.Tensor:
        pred_out = []
        for beam in self._decoding_state.beams:
            pred_out.append(beam.pred_out)
        return torch.cat(pred_out, dim=0)

    def decode(self, hidden_states: torch.Tensor) -> str:
        # Rnnt Beam decoding, Mainly for eval. max_token_step will be restricted as 1.
        # hidden_states: torch.Tensor(1, seq_len, dim)
        assert hidden_states.shape[0] == 1, "Support BatchSize = 1 only."

        self.reset(hidden_states.device)  # Reset decoding state

        tot_time_steps = hidden_states.shape[1]
        curr_time_step = 0

        while curr_time_step < tot_time_steps:
            # After consuming all hidden_state, decoding session end.
            enc_out = hidden_states[:, curr_time_step:curr_time_step +
                                    1, :]  #（1, 1, D）

            log_probs = self._joiner.streaming_step(
                enc_out, self._build_beam_pred_out())  # (self._beam_size, D)

            self._update_beams(log_probs)

            # Update pred_state and pred_out corresponding to beam within pruned beams
            for beam_id in range(len(self._decoding_state.beams)):
                beam = self._decoding_state.beams[beam_id]

                if not beam.end_with_blank:
                    # Forward predictor for only 1 step, update pred_state and pred_out of given
                    # beam, given condition that each time step emit no more one token in this
                    # decoding impl.
                    pred_out, pred_state = self._predictor.streaming_step(
                        torch.tensor([[beam.decoded_tokens[-1]]
                                     ]).long().to(hidden_states.device),
                        beam.pred_state)
                    beam.end_with_blank = True
                    beam.pred_state = pred_state
                    beam.pred_out = pred_out
            curr_time_step += 1

        return self._tokenizer.decode(
            torch.Tensor(self._decoding_state.best_beam.decoded_tokens).long())

    def _update_beams(self, log_probs: torch.Tensor):
        # Update decoding state with log_probs of current time step.
        new_beams = []
        for beam_id in range(len(self._decoding_state.beams)):
            beam = self._decoding_state.beams[beam_id]

            token_idxs = torch.argsort(log_probs[beam_id],
                                       descending=True).tolist()

            for token_id in token_idxs[:self._cutoff_top_k]:
                if token_id == 0:
                    # If is <blank_id>, update beam with pred_out and pred_state indicating
                    # this beam would not update at this timestep.
                    new_beams.append(
                        DecodedBeam(decoded_tokens=beam.decoded_tokens,
                                    end_with_blank=True,
                                    score=beam.score +
                                    log_probs[beam_id][token_id],
                                    pred_state=beam.pred_state,
                                    pred_out=beam.pred_out))
                if token_id != 0:
                    # If not <blank_id>, update Beams with new beam, new beam will update pred state
                    # only, cos at end of this iteration of this time step predictor would move forward
                    # one step on token direction, which will update pred_state and pred_out.
                    new_beams.append(
                        DecodedBeam(
                            decoded_tokens=beam.decoded_tokens + [token_id],
                            end_with_blank=False,
                            score=beam.score + log_probs[beam_id][token_id],
                            pred_state=beam.pred_state))
        # Prune beam
        self._decoding_state.beams = sorted(new_beams,
                                            key=lambda x: x.score,
                                            reverse=True)[:self._beam_size]
        # Best beam
        self._decoding_state.best_beam = self._decoding_state.beams[0]


@unique
class DecodingFactory(Enum):
    """ Decoding factory for task setting """
    ctc_greedy_decoding = CtcGreedyDecoding
    ctc_lexicon_beam_decoding = CtcLexiconBeamDecoding
    cif_greedy_decoding = CifGreedyDecoding
    rnnt_greedy_decoding = RnntGreedyDecoding
    rnnt_beam_decoding = RnntBeamDecoding
