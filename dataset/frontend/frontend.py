# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.07
""" Frontend of wav processing """

import torchaudio
import torch
import torch.nn as nn


class DummyFrontend(nn.Module):
    """ Dummy Frontend to comply with AsrTask API, based on Wav2Vec2
        model is directly trained on Pcms instead of acoustic feats
        like MFCC or Fbank. 
    """

    def __init__(self, dummy=-1) -> None:
        super(DummyFrontend, self).__init__()
        self._dummy = dummy

    @torch.no_grad()
    def forward(self, pcm: torch.Tensor) -> torch.Tensor:
        return pcm.squeeze(0)


class KaldiWaveFeature(nn.Module):
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
