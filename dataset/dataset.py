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

import dataset.frontend.data_augmentation as data_augmentation
import dataset.utils as utils
from dataset.frontend.frontend import KaldiWaveFeature, DummyFrontend


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
        return datamap

    def _make_noiseset_from_json(self, noise_json):
        # Make Noise datapool for add noise data_augmentation
        with open(noise_json, 'r') as json_f:
            for line in json_f:
                data_infos = json.loads(line)
                self._noise_dataset.append(data_infos["noise_filepath"])

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

        self._feat_type = config["feat_type"]
        self._data_aug_config = config["data_aug_config"]
        self._tokenizer = tokenizer

        # Add Noise data_augmentation
        self._add_noise_proportion = self._data_aug_config[
            "add_noise_proportion"]
        self._add_noise_config = self._data_aug_config["add_noise_config"]
        self._add_noise = data_augmentation.add_noise

        # Speed_perturb augmentation
        self._speed_perturb = data_augmentation.speed_perturb

        # Spec_augmentation
        self._spec_augment = data_augmentation.spec_aug

        if self._feat_type == "fbank":
            self._compute_feature = KaldiWaveFeature(**config["feat_config"])
        elif self._feat_type == "pcm":
            self._compute_feature = DummyFrontend(**config["feat_config"])
        else:
            raise ValueError(
                "feat_type only support 'fbank' and 'pcm' right now, please check your config."
            )

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

        pcm, framerate = torchaudio.load(data["audio_filepath"], normalize=True)

        # Data Augmentation: Add Noise
        # Use add noise proportion control the augmentation ratio of all dataset
        need_noisify_aug = random.uniform(0, 1) < self._add_noise_proportion
        if need_noisify_aug:
            noise_pcm, _ = torchaudio.load(random.choice(self._noise_dataset),
                                           normalize=True)
            pcm = self._add_noise(pcm, noise_pcm, **self._add_noise_config)

        if self._data_aug_config["use_speed_perturb"]:
            pcm = self._speed_perturb(pcm)  # Speed_perturb aug

        feat = self._compute_feature(pcm)  # Extract acoustic feats

        if self._feat_type == "fbank" and self._data_aug_config["use_spec_aug"]:
            feat = self._spec_augment(feat)  # Spec_aug

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

        self._feat_type = config["feat_type"]
        self._tokenizer = tokenizer

        if self._feat_type == "fbank":
            self._compute_feature = KaldiWaveFeature(**config["feat_config"])
        elif self._feat_type == "pcm":
            self._compute_feature = DummyFrontend(**config["feat_config"])
        else:
            raise ValueError(
                "feat_type only support 'fbank' and 'pcm' right now, please check your config."
            )

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

        pcm, framerate = torchaudio.load(data["audio_filepath"], normalize=True)
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
            dataset_json,
            frontend,
            dur_min_filter=0.0,
            dur_max_filter=float("inf"),
    ) -> None:
        # Testset should not filter any of the data, set infinite as dur_max_filter factor as deflaut
        super(AsrTestDataset, self).__init__(
            dataset_json=dataset_json,
            dur_min_filter=dur_min_filter,
            dur_max_filter=dur_max_filter,
        )

        # Load TorchScript frontend graph to initialize featrue extraction session
        self._frontend_sess = torch.jit.load(frontend)

    def __getitem__(self, index):
        """ Return:
                {"feat": Tensor.float(T, D),
                 "text": str}
        """
        data = self._dataset[index]

        assert "audio_filepath" in data
        assert "text" in data

        pcm, framerate = torchaudio.load(data["audio_filepath"], normalize=True)
        feat = self._frontend_sess(pcm)

        return {"feat": feat, "text": data["text"]}


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


class AsrSampler(Sampler):
    # TODO: customized desgin sampler or distributed sampler would work?
    def __init__(self, data_source):
        super(AsrSampler, self).__init__(data_source=data_source)

    def __iter__(self):
        pass
