# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.07
""" Frontend of wav processing """

import abc
import torchaudio
import torch
import torch.nn as nn

from enum import Enum, unique
from lhotse import KaldifeatFbank, KaldifeatFbankConfig


class FeaturePipeline(abc.ABC, nn.Module):
    """ Abstract class for FeaturePipeline """

    @property
    @abc.abstractmethod
    def pcm_normalize(self):
        pass

    @property
    @abc.abstractmethod
    def feat_dim(self):
        pass

    @torch.no_grad()
    @abc.abstractmethod
    def forward(self, pcm: torch.Tensor) -> torch.Tensor:
        pass


class DummyFrontend(FeaturePipeline):
    """ Dummy Frontend to comply with AsrTask API, based on Wav2Vec2
        model is directly trained on Pcms instead of acoustic feats
        like MFCC or Fbank. 
    """

    def __init__(self, dummy=-1) -> None:
        super(DummyFrontend, self).__init__()
        self._dummy = dummy

    @property
    def pcm_normalize(self):
        return True

    @property
    def feat_dim(self):
        return -1

    @torch.no_grad()
    def forward(self, pcm: torch.Tensor) -> torch.Tensor:
        return pcm.squeeze(0)


class KaldiWaveFeature(FeaturePipeline):
    """ Frontend of train, eval and test dataset to get feature from
        PCMs in kaldi-style
    """

    def __init__(self,
                 num_mel_bins=64,
                 frame_length=25,
                 frame_shift=10,
                 dither=0.0,
                 samplerate=16000) -> None:
        super(KaldiWaveFeature, self).__init__()

        self._feature_extractor = torchaudio.compliance.kaldi.fbank
        self._num_mel_bins = num_mel_bins
        self._frame_length = frame_length
        self._frame_shift = frame_shift
        self._dither = dither
        self._samplerate = samplerate

    @property
    def pcm_normalize(self):
        return True

    @property
    def feat_dim(self):
        return self._num_mel_bins

    @torch.no_grad()
    def forward(self, pcm: torch.Tensor) -> torch.Tensor:
        features = self._feature_extractor(pcm,
                                           num_mel_bins=self._num_mel_bins,
                                           frame_length=self._frame_length,
                                           frame_shift=self._frame_shift,
                                           dither=self._dither,
                                           energy_floor=0.0,
                                           sample_frequency=self._samplerate)
        return features


class LhotseKaldiFeatFbank(nn.Module):
    """ Feature pipeline used by icefall """

    def __init__(self, num_mel_bins=80) -> None:
        super(LhotseKaldiFeatFbank, self).__init__()

        # Device will always on cpu
        self._num_mel_bins = num_mel_bins
        self._extractor = KaldifeatFbank(config=KaldifeatFbankConfig(
            device="cpu"))

    @property
    def pcm_normalize(self):
        return True

    @property
    def feat_dim(self):
        return self._num_mel_bins

    def forward(self, pcm: torch.Tensor) -> torch.Tensor:
        return self._extractor.extract(samples=pcm, sampling_rate=16000)


@unique
class FeatType(Enum):
    """ Feature pipeline factory """
    pcm = DummyFrontend
    fbank = KaldiWaveFeature
    lhotes_fbank = LhotseKaldiFeatFbank
