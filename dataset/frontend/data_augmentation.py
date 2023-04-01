# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.09
""" Implementation of data augmention
    Modified from wenet/dataset/processor.py
"""

import random
import torch
import torchaudio


class NoiseProcessor(object):
    """ Interface of unitities of add_noise, Borrowed from 
        nemo.collections.asr.parts.preprocessing.segment 
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def rms_db(pcm: torch.Tensor):
        # pcm is torch.Tensor get through torchaudio.load
        mean_square = (pcm**2).mean()
        return 10 * torch.log10(mean_square)

    @staticmethod
    def gain_db(pcm, gain):
        pcm *= 10.0**(gain / 20.0)
        return pcm


def add_noise(pcm: torch.Tensor,
              noise_pcm: torch.Tensor,
              min_snr_db=10,
              max_snr_db=50,
              max_gain_db=300.0):
    """ Add noise if 'noise_pcm' field provided, the ratio of noise augment control
        has been relocated to outside of func
    """
    snr_db = random.uniform(min_snr_db, max_snr_db)
    data_rms = NoiseProcessor.rms_db(pcm)
    noise_rms = NoiseProcessor.rms_db(noise_pcm)
    noise_gain_db = min(data_rms - noise_rms - snr_db, max_gain_db)

    noise_pcm = NoiseProcessor.gain_db(noise_pcm, noise_gain_db)
    if pcm.shape[1] > noise_pcm.shape[1]:
        noise_pcm = noise_pcm.repeat(
            1,
            torch.div(pcm.shape[1], noise_pcm.shape[1], rounding_mode="floor") +
            1)
    return pcm + noise_pcm[:, :pcm.shape[1]]


def speed_perturb(pcm: torch.Tensor,
                  sample_rate=16000,
                  min_speed=0.9,
                  max_speed=1.1,
                  rate=3):
    """ Apply speed perturb to the data.
        Inplace operation.
    """
    speeds = torch.linspace(min_speed, max_speed, steps=rate).tolist()
    speed = random.choice(speeds)
    if speed != 1.0:
        perturbed_pcm, _ = torchaudio.sox_effects.apply_effects_tensor(
            pcm, sample_rate,
            [['speed', str(speed)], ['rate', str(sample_rate)]])
        return perturbed_pcm
    else:
        return pcm


def spec_aug(feat: torch.Tensor,
             num_t_mask=2,
             num_f_mask=2,
             max_t=50,
             max_f=10,
             max_w=80):
    """ Do spec augmentation
        Inplace operation
        Args:
            feat: torch.Tensor(T, D) 
            num_t_mask: number of time mask to apply 
            num_f_mask: number of freq mask to apply 
            max_t: max width of time mask 
            max_f: max width of freq mask 
            max_w: max width of time warp
        Returns
            spec_auged_feat: torch.Tensor(T, D)
    """
    assert isinstance(feat, torch.Tensor)
    y = feat.clone().detach()
    max_frames = y.size(0)
    max_freq = y.size(1)

    # time mask
    for i in range(num_t_mask):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        y[start:end, :] = 0

    # freq mask
    for i in range(num_f_mask):
        start = random.randint(0, max_freq - 1)
        length = random.randint(1, max_f)
        end = min(max_freq, start + length)
        y[:, start:end] = 0
    return y
