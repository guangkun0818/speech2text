# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.09
""" Implementation of data augmention
    Modified from wenet/dataset/processor.py
"""

import random
import torch
import torchaudio


class AddNoise(object):

    def __init__(self, min_snr_db=10, max_snr_db=50, max_gain_db=300.0) -> None:
        self._min_snr_db = min_snr_db
        self._max_snr_db = max_snr_db
        self._max_gain_db = max_gain_db

    @staticmethod
    def rms_db(pcm: torch.Tensor):
        # pcm is torch.Tensor get through torchaudio.load
        mean_square = (pcm**2).mean()
        return 10 * torch.log10(mean_square)

    @staticmethod
    def gain_db(pcm, gain):
        pcm *= 10.0**(gain / 20.0)
        return pcm

    def process(self, pcm: torch.Tensor,
                noise_pcm: torch.Tensor) -> torch.Tensor:
        # Add noise on torch.float tensor
        if pcm.dtype != torch.float:
            pcm = pcm.float()
        if noise_pcm.dtype != torch.float:
            noise_pcm = pcm.float()

        snr_db = random.uniform(self._min_snr_db, self._max_snr_db)
        data_rms = self.rms_db(pcm)
        noise_rms = self.rms_db(noise_pcm)
        noise_gain_db = min(data_rms - noise_rms - snr_db, self._max_gain_db)

        noise_pcm = self.gain_db(noise_pcm, noise_gain_db)
        if pcm.shape[1] > noise_pcm.shape[1]:
            noise_pcm = noise_pcm.repeat(
                1,
                torch.div(
                    pcm.shape[1], noise_pcm.shape[1], rounding_mode="floor") +
                1)
        # Random pick start and end timestamp of noise_pcm
        start = random.randint(0, noise_pcm.shape[1] - pcm.shape[1])
        end = start + pcm.shape[1]
        auged_pcm = pcm + noise_pcm[:, start:end]
        auged_pcm = torch.clip(auged_pcm, min=-1.0, max=1.0)
        return auged_pcm


class MixFeats(object):
    """ Mix Audio feats with noise feats based on SNR setting, borrow the ideas
        from icefall.
        
        Args:
            src: src feats to be mixed. (T, D) 
            noise: noise feats to mix. (T, D) 
            snrs: Range bewteen min_snr and max_snr
        return
            mixed_feats
    """

    def __init__(self, snrs=(10, 20)) -> None:
        self._snrs = snrs

    @staticmethod
    def compute_energy(feats: torch.Tensor) -> float:
        return float(torch.sum(torch.exp(feats)))

    @staticmethod
    def compute_gain(src_energy: float, noise_energy: float, snr: float):
        gain = 1.0
        if src_energy > 0.0:
            # Compute the added signal energy before it was padded
            added_feats_energy = noise_energy
            if added_feats_energy > 0.0:
                target_energy = src_energy * (10.0**(-snr / 10))
                gain = target_energy / added_feats_energy
        return gain

    @staticmethod
    def mix(features_a: torch.Tensor, features_b: torch.Tensor,
            energy_scaling_factor_b: float) -> torch.Tensor:
        return torch.log(
            torch.clip(
                torch.exp(features_a) +
                energy_scaling_factor_b * torch.exp(features_b),
                # protection against 1og(0); max with EPSILON is adequate since these are energies (always >= 0)
                min=1e-10,  # EPSILON = 1e-10
            ))

    def process(self, src: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:

        src_energy = self.compute_energy(feats=src)
        noise_energy = self.compute_energy(feats=noise)
        snr = random.uniform(self._snrs[0], self._snrs[-1])

        gain = self.compute_gain(src_energy, noise_energy, snr)

        # Pad on noise_feats if it short than src_feats, same as add_noise
        if src.shape[0] > noise.shape[0]:
            noise = noise.repeat(
                torch.div(src.shape[0], noise.shape[0], rounding_mode="floor") +
                1, 1)

        # Random pick start and end timestamp of noise pem.
        start = random.randint(0, noise.shape[0] - src.shape[0])
        end = start + src.shape[0]
        auged_feats = self.mix(src, noise[start:end, :], gain)
        return auged_feats


class SpeedPerturb(object):
    """ Apply speed perturb to the data.
        Inplace operation.
    """

    def __init__(self,
                 sample_rate=16000,
                 min_speed=0.9,
                 max_speed=1.1,
                 rate=3) -> None:
        self._sample_rate = sample_rate
        self._min_speed = min_speed
        self._max_speed = max_speed
        self._rate = rate

    def process(self, pcm: torch.Tensor) -> torch.Tensor:
        speeds = torch.linspace(self._min_speed,
                                self._max_speed,
                                steps=self._rate).tolist()
        speed = random.choice(speeds)
        if speed != 1.0:
            perturbed_pcm, _ = torchaudio.sox_effects.apply_effects_tensor(
                pcm, self._sample_rate,
                [['speed', str(speed)], ['rate', str(self._sample_rate)]])
            return perturbed_pcm
        else:
            return pcm


class SpecAugment(object):
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

    def __init__(self,
                 num_t_mask=2,
                 num_f_mask=2,
                 max_t=50,
                 max_f=10,
                 max_w=80) -> None:

        self._num_t_mask = num_t_mask
        self._num_f_mask = num_f_mask
        self._max_t = max_t
        self._max_f = max_f
        self._max_w = max_w

    def process(self, feat: torch.Tensor) -> torch.Tensor:
        assert isinstance(feat, torch.Tensor)
        y = feat.clone().detach()
        max_frames = y.size(0)
        max_freq = y.size(1)

        # time mask
        for i in range(self._num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, self._max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0

        # freq mask
        for i in range(self._num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, self._max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        return y
