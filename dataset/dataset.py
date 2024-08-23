# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.03.29
""" Make Torch Dataset for Asr task or more """

import abc
import glog
import json
import random
import torch
import torchaudio

from typing import Dict, List
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence

import dataset.frontend.data_augmentation as data_augmentation
import dataset.utils as utils
from dataset.frontend.frontend import FeatType


class BaseDataset(Dataset):
    """ Base ASR Dataset to inherit for train, eval, test """

    def __init__(self,
                 dataset_json,
                 dur_min_filter=0.0,
                 dur_max_filter=20.0,
                 noiseset_json=None) -> None:
        """ Args:
                dataset_json: JSON file of train data, same as NeMo.
                dur_filter: Use JSON duration of dataset to filter dataset
                noiseset_json: JSON file of noise data
        """
        super(BaseDataset, self).__init__()
        self._total_duration = 0.0
        self._min_duration = float("inf")
        self._max_duration = -float("inf")
        self._dataset = self._make_dataset_from_json(dataset_json,
                                                     dur_min_filter,
                                                     dur_max_filter)

        # Load noise set if add noise applied
        self._noise_dataset = []
        if noiseset_json is not None:
            self._make_noiseset_from_json(noiseset_json)

    def _make_dataset_from_json(self, json_file, dur_min_filter,
                                dur_max_filter):
        """ Make Dataset list from JSON file """
        datamap = []
        with open(json_file, 'r') as json_f:
            for line in json_f:
                data_infos = json.loads(line)
                if dur_min_filter <= data_infos["duration"] <= dur_max_filter:
                    datamap.append(data_infos)
                    self._total_duration += data_infos["duration"]
                    self._min_duration = min(self._min_duration,
                                             data_infos["duration"])
                    self._max_duration = max(self._max_duration,
                                             data_infos["duration"])
        return datamap

    def _make_noiseset_from_json(self, noise_json):
        # Make Noise datapool for add noise data_augmentation
        with open(noise_json, 'r') as json_f:
            for line in json_f:
                data_infos = json.loads(line)
                self._noise_dataset.append(data_infos)

    def fetch_data_k_info(self, idx, k):
        """ Fetch data info with key and idx, for bucket sampling.
            Args:
                idx: int, Original dataset index
                key: str, info key within single data entry
            return:
                Any.
        """
        return self._dataset[idx][k]

    def compute_offset(self, start: float, end: float, frame_rate=16000):
        # Compute audio segment offset.
        frame_offset = int(start * frame_rate)
        num_frames = int(end * frame_rate) - frame_offset
        return frame_offset, num_frames

    @property
    def min_duration(self):
        return self._min_duration

    @property
    def max_duration(self):
        return self._max_duration

    @property
    def total_duration(self):
        return self._total_duration

    def __len__(self):
        """ Overwrite __len__ """
        return len(self._dataset)

    @abc.abstractmethod
    def __getitem__(self, index):
        """ Implement for train, eval, test specifically """
        pass


class AsrTrainDataset(BaseDataset):
    """ ASR TrainDataset with Data Augementation"""

    def __init__(self, config, tokenizer: utils.TokenizerSetup) -> None:
        super(AsrTrainDataset,
              self).__init__(dataset_json=config["train_data"],
                             dur_min_filter=config["dur_min_filter"],
                             dur_max_filter=config["dur_max_filter"],
                             noiseset_json=config["noise_data"])

        glog.info("Train dataset duration: {}h.".format(
            self.total_duration / 3600, ".2f"))

        self._dataset_config = config
        self._tokenizer = tokenizer
        self._compute_feature = FeatType[config["feat_type"]].value(
            **config["feat_config"])

        # Add Noise data_augmentation
        self._data_aug_config = config["data_aug_config"]
        self._add_noise_proportion = self._data_aug_config[
            "add_noise_proportion"]
        self._add_noise = data_augmentation.AddNoise(
            **self._data_aug_config["add_noise_config"])
        self._speed_perturb = data_augmentation.SpeedPerturb()
        self._spec_augment = data_augmentation.SpecAugment()
        self._mix_feats_proportion = self._data_aug_config[
            "mix_feats_proportion"]
        self._mix_feats = data_augmentation.MixFeats(
            **self._data_aug_config["mix_feats_config"])

    def __getitem__(self, index):
        """ Return:
                {"feat": Tensor.float(T, D),
                 "feat_length": int,
                 "label": Tensor.long(U),
                 "label_length": int}
        """
        data = self._dataset[index]

        assert "audio_filepath" in data
        assert "text" in data

        # Apply segmentation on origin audio, quite slow not recommend.

        frame_offset, num_frames = self.compute_offset(
            start=data["segment"][0], end=data["segment"]
            [1]) if self._dataset_config["apply_segment"] else 0, -1

        pcm, framerate = torchaudio.load(
            data["audio_filepath"],
            frame_offset=frame_offset,
            num_frames=num_frames,
            normalize=self._compute_feature.pcm_normalize)

        # Data Augmentation: Add Noise
        if self._data_aug_config["use_add_noise"]:
            # Use add noise proportion control the augmentation ratio of all dataset
            need_noisify_aug = random.uniform(0, 1) < self._add_noise_proportion
            if need_noisify_aug:
                noise_pcm, _ = torchaudio.load(
                    random.choice(self._noise_dataset)["noise_filepath"],
                    normalize=self._compute_feature.pcm_normalize)
                pcm = self._add_noise.process(pcm, noise_pcm)

        if self._data_aug_config["use_speed_perturb"]:
            pcm = self._speed_perturb.process(pcm)  # Speed_perturb aug

        feat = self._compute_feature(pcm)  # Extract acoustic feats

        if self._data_aug_config["use_mix_feats"]:
            need_mix_feats = random.uniform(0, 1) < self._mix_feats_proportion
            if need_mix_feats:
                noise_entry = random.choice(self._noise_dataset)
                # Avoid waste on compute feats on unused noise_pcm.
                start_t = random.uniform(
                    0, max(0, noise_entry["duration"] - data["duration"]))
                end_t = min(start_t + data["duration"], noise_entry["duration"])
                frame_offset, num_frames = self.compute_offset(start_t, end_t)

                noise_pcm, _ = torchaudio.load(
                    noise_entry["noise_filepath"],
                    frame_offset=frame_offset,
                    num_frames=num_frames,
                    normalize=self._compute_feature.pcm_normalize)
                noise_feats = self._compute_feature(noise_pcm)
                feat = self._mix_feats.process(src=feat, noise=noise_feats)

        if self._data_aug_config["use_spec_aug"]:
            feat = self._spec_augment.process(feat)  # Spec_aug

        label_tensor = self._tokenizer.encode(data["text"])

        return {
            "feat": feat,  # (T, D)
            "feat_length": feat.shape[0],
            "label": label_tensor,  # (U)
            "label_length": label_tensor.shape[0]
        }


class AsrEvalDataset(BaseDataset):
    """ ASR EvalDataset without Data Augementation """

    def __init__(self, config, tokenizer: utils.TokenizerSetup) -> None:
        super(AsrEvalDataset,
              self).__init__(dataset_json=config["eval_data"],
                             dur_min_filter=config["dur_min_filter"],
                             dur_max_filter=config["dur_max_filter"],
                             noiseset_json=None)

        glog.info("Eval dataset duration: {}h.".format(
            self.total_duration / 3600, ".2f"))

        self._dataset_config = config
        self._tokenizer = tokenizer
        self._compute_feature = FeatType[config["feat_type"]].value(
            **config["feat_config"])

    def __getitem__(self, index):
        """ Return:
                {"feat": Tensor.float(T, D),
                 "feat_length": int,
                 "label": Tensor.long(U),
                 "label_length": int}
        """
        data = self._dataset[index]

        assert "audio_filepath" in data
        assert "text" in data

        # Apply segmentation on origin audio, quite slow not recommend.
        frame_offset, num_frames = self.compute_offset(
            start=data["segment"][0], end=data["segment"]
            [1]) if self._dataset_config["apply_segment"] else 0, -1

        pcm, framerate = torchaudio.load(
            data["audio_filepath"],
            frame_offset=frame_offset,
            num_frames=num_frames,
            normalize=self._compute_feature.pcm_normalize)

        feat = self._compute_feature(pcm)

        label_tensor = self._tokenizer.encode(data["text"])

        return {
            "feat": feat,  # (T, D)
            "feat_length": feat.shape[0],
            "label": label_tensor,  # (U)
            "label_length": label_tensor.shape[0]
        }


class AsrTestDataset(BaseDataset):
    """ Asr TestDataset without Data Augementation and batching """

    def __init__(
            self,
            testset_json,
            testset_config,
            dur_min_filter=0.0,
            dur_max_filter=float("inf"),
    ) -> None:
        # Testset should not filter any of the data, set infinite as dur_max_filter factor as deflaut
        super(AsrTestDataset, self).__init__(dataset_json=testset_json,
                                             dur_min_filter=dur_min_filter,
                                             dur_max_filter=dur_max_filter,
                                             noiseset_json=None)

        glog.info("Test dataset duration: {}h.".format(
            self.total_duration / 3600, ".2f"))

        self._testset_config = testset_config
        self._compute_feature = FeatType[testset_config["feat_type"]].value(
            **testset_config["feat_config"])

    def __getitem__(self, index):
        """ Return:
                {"feat": Tensor.float(T, D),
                 "text": str}
        """
        data = self._dataset[index]

        assert "audio_filepath" in data
        assert "text" in data

        # Apply segmentation on origin audio, quite slow not recommend.
        frame_offset, num_frames = self.compute_offset(
            start=data["segment"][0], end=data["segment"]
            [1]) if self._testset_config["apply_segment"] else 0, -1

        pcm, framerate = torchaudio.load(
            data["audio_filepath"],
            frame_offset=frame_offset,
            num_frames=num_frames,
            normalize=self._compute_feature.pcm_normalize)

        feat = self._compute_feature(pcm)

        return {
            "audio_filepath": data["audio_filepath"],
            "feat": feat,
            "text": data["text"]
        }


def asr_test_collate_fn(raw_batch: List[Dict]) -> Dict:
    """ Batching and Padding sequence right before output, 
        implement for train, eval 
    """
    batch = {
        "audio_filepath": [],
        "feat": [],
        "text": [],
    }
    for data_slice in raw_batch:
        # Reorganize batch data as Map

        batch["audio_filepath"].append(data_slice["audio_filepath"])
        batch["feat"].append(data_slice["feat"])
        batch["text"].append(data_slice["text"])

    batch["feat"] = pad_sequence(batch["feat"],
                                 batch_first=True,
                                 padding_value=0)
    return batch


class SslTrainDataset(BaseDataset):
    """ Self-supervised learning train dataset 
        
        Ssl Feature Pipeline: 
        load_pcm() -> speed_perturb() -> compute_feats() -> raw_feats
                            ↓
                        add_noise() -> mix_feats() -> compute_feats() -> spec_aug() -> auged_feats
    """

    def __init__(self, config) -> None:
        super(SslTrainDataset,
              self).__init__(dataset_json=config["train_data"],
                             dur_min_filter=config["dur_min_filter"],
                             dur_max_filter=config["dur_max_filter"],
                             noiseset_json=config["noise_data"])

        glog.info("Train dataset duration: {}h.".format(
            self.total_duration / 3600, ".2f"))

        self._dataset_config = config
        self._compute_feature = FeatType[config["feat_type"]].value(
            **config["feat_config"])

        # Add Noise data_augmentation
        self._data_aug_config = config["data_aug_config"]
        self._add_noise_proportion = self._data_aug_config[
            "add_noise_proportion"]
        self._add_noise = data_augmentation.AddNoise(
            **self._data_aug_config["add_noise_config"])
        self._speed_perturb = data_augmentation.SpeedPerturb()
        self._spec_augment = data_augmentation.SpecAugment()
        self._mix_feats_proportion = self._data_aug_config[
            "mix_feats_proportion"]
        self._mix_feats = data_augmentation.MixFeats(
            **self._data_aug_config["mix_feats_config"])

    def __getitem__(self, index):
        """ Return: {
                "raw_feat": Tensor.float(T, D),
                "auged_feat": Tensor.float(T, D),
                "feat_length": int
            }
        """
        data = self._dataset[index]

        assert "audio_filepath" in data
        assert "text" in data

        # Apply segmentation on origin audio, quite slow not recommend.

        frame_offset, num_frames = self.compute_offset(
            start=data["segment"][0], end=data["segment"]
            [1]) if self._dataset_config["apply_segment"] else 0, -1

        raw_pcm, framerate = torchaudio.load(
            data["audio_filepath"],
            frame_offset=frame_offset,
            num_frames=num_frames,
            normalize=self._compute_feature.pcm_normalize)

        if self._data_aug_config["use_speed_perturb"]:
            raw_pcm = self._speed_perturb.process(
                raw_pcm)  # Speed_perturb on raw pcm

        raw_feat = self._compute_feature(
            raw_pcm)  # Extract acoustic feats of raw pcm

        # Data Augmentation: Add Noise
        auged_pcm = raw_pcm
        if self._data_aug_config["use_add_noise"]:
            # Use add noise proportion control the augmentation ratio of all dataset
            need_noisify_aug = random.uniform(0, 1) < self._add_noise_proportion
            if need_noisify_aug:
                noise_pcm, _ = torchaudio.load(
                    random.choice(self._noise_dataset)["noise_filepath"],
                    normalize=self._compute_feature.pcm_normalize)
                auged_pcm = self._add_noise.process(raw_pcm, noise_pcm)

        auged_feat = self._compute_feature(auged_pcm)  # Extract acoustic feats

        if self._data_aug_config["use_mix_feats"]:
            need_mix_feats = random.uniform(0, 1) < self._mix_feats_proportion
            if need_mix_feats:
                noise_entry = random.choice(self._noise_dataset)
                # Avoid waste on compute feats on unused noise_pcm.
                start_t = random.uniform(
                    0, max(0, noise_entry["duration"] - data["duration"]))
                end_t = min(start_t + data["duration"], noise_entry["duration"])
                frame_offset, num_frames = self.compute_offset(start_t, end_t)

                noise_pcm, _ = torchaudio.load(
                    noise_entry["noise_filepath"],
                    frame_offset=frame_offset,
                    num_frames=num_frames,
                    normalize=self._compute_feature.pcm_normalize)
                noise_feats = self._compute_feature(noise_pcm)
                auged_feat = self._mix_feats.process(src=auged_feat,
                                                     noise=noise_feats)

        if self._data_aug_config["use_spec_aug"]:
            auged_feat = self._spec_augment.process(auged_feat)  # Spec_aug

        return {
            "raw_feat": raw_feat,  # (T, D)
            "auged_feat": auged_feat,
            "feat_length": raw_feat.shape[0],
        }


class SslEvalDataset(BaseDataset):
    """ Self-supervised learning eval dataset """

    def __init__(self, config) -> None:
        super(SslEvalDataset,
              self).__init__(dataset_json=config["eval_data"],
                             dur_min_filter=config["dur_min_filter"],
                             dur_max_filter=config["dur_max_filter"],
                             noiseset_json=None)
        glog.info("Eval dataset duration: {}h.".format(
            self.total_duration / 3600, ".2f"))

        self._dataset_config = config
        self._compute_feature = FeatType[config["feat_type"]].value(
            **config["feat_config"])

    def __getitem__(self, index):
        """ Return: {
                "raw_feat": Tensor.float(T, D),
                "auged_feat": Tensor.float(T, D),
                "feat_length": int
            }
        """
        data = self._dataset[index]

        assert "audio_filepath" in data
        assert "text" in data

        # Apply segmentation on origin audio, quite slow not recommend.
        frame_offset, num_frames = self.compute_offset(
            start=data["segment"][0], end=data["segment"]
            [1]) if self._dataset_config["apply_segment"] else 0, -1

        pcm, framerate = torchaudio.load(
            data["audio_filepath"],
            frame_offset=frame_offset,
            num_frames=num_frames,
            normalize=self._compute_feature.pcm_normalize)

        feat = self._compute_feature(pcm)

        return {
            "raw_feat": feat,
            "auged_feat": feat,
            "feat_length": feat.shape[0],
        }


def asr_collate_fn(raw_batch: List[Dict]) -> Dict:
    """ Batching and Padding sequence right before output, 
        implement for train, eval 
    """
    batch_map = {
        "feat": [],
        "feat_length": [],
        "label": [],
        "label_length": [],
    }
    for data_slice in raw_batch:
        # Reorganize batch data as Map
        glog.check("feat" in data_slice.keys())
        glog.check("feat_length" in data_slice.keys())
        glog.check("label" in data_slice.keys())
        glog.check("label_length" in data_slice.keys())

        batch_map["feat"].append(data_slice["feat"])
        batch_map["feat_length"].append(data_slice["feat_length"])
        batch_map["label"].append(data_slice["label"])
        batch_map["label_length"].append(data_slice["label_length"])

    batch = utils.batch(batch_map)
    return batch


def ssl_collate_fn(raw_batch: List[Dict]) -> Dict:
    """ Batching and Padding sequence right before output, 
        implement self-supervised learning.
    """
    batch_map = {
        "raw_feat": [],
        "auged_feat": [],
        "feat_length": [],
    }
    for data_slice in raw_batch:
        # Reorganize batch data as Map
        glog.check("raw_feat" in data_slice.keys())
        glog.check("auged_feat" in data_slice.keys())
        glog.check("feat_length" in data_slice.keys())

        batch_map["raw_feat"].append(data_slice["raw_feat"])
        batch_map["auged_feat"].append(data_slice["auged_feat"])
        batch_map["feat_length"].append(data_slice["feat_length"])

    batch = utils.batch(batch_map)
    return batch


class AsrSampler(Sampler):
    # TODO: customized desgin sampler or distributed sampler would work?
    def __init__(self, data_source):
        super(AsrSampler, self).__init__(data_source=data_source)

    def __iter__(self):
        pass
