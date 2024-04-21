# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.04.21
""" Pre-processing of building a training task, train a spm model for 
    tokenizer setup. 
"""

import os
import dataclasses
import json
import glog
import shutil
import sentencepiece as spm


@dataclasses.dataclass
class SpmTrainConfig:
    """ SentencePiece model training config """
    vocab_size: int = None
    model_type: str = None
    spm_export_path: str = None


class SpmTrain(object):
    """ Train a SentencePiece model if Subword tokenizer specified but no 
        Spm model provided. 
    """

    def __init__(self, config: SpmTrainConfig) -> None:
        super().__init__()
        self._vocab_size = config.vocab_size
        self._model_type = config.model_type
        self._spm_export_path = config.spm_export_path

    def train(self, train_data) -> None:
        self._train_spm(dataset=train_data)

    def _train_spm(self, dataset):
        # Train SenetencePiece model base on train_data corpus
        os.makedirs(self._spm_export_path, exist_ok=True)
        with open(os.path.join(self._spm_export_path, "corpus"),
                  'w') as corpus_f, open(dataset, 'r') as data_json:
            for line in data_json:
                data_infos = json.loads(line)
                corpus_f.write("{}\n".format(data_infos["text"]))

        spm.SentencePieceTrainer.Train("--input={} \
            --vocab_size={} \
            --character_converage=1 \
            --model_type={} \
            --model_prefix={} \
            --input_sentence_size=100000000".format(
            os.path.join(self._spm_export_path, "corpus"), self._vocab_size,
            self._model_type, os.path.join(self._spm_export_path, "subword")))


def _spm_training(config):
    """ Ineternal factory function for spm training. """

    spm_train_config = SpmTrainConfig(**config["tokenizer"]["train_config"])

    if config["resume"] is not None:
        assert os.path.exists(
            os.path.join(spm_train_config.spm_export_path, "subword.model"))
        assert os.path.exists(
            os.path.join(spm_train_config.spm_export_path, "subword.vocab"))
        glog.info(
            "`spm_train` will be overrided since `resume` is set, use existed one."
        )
        return
    else:
        glog.info("Training spm model with config: {}".format(
            config["tokenizer"]["train_config"]))
        spm_trainer = SpmTrain(config=spm_train_config)
        train_data = config["dataset"]["train_data"]
        spm_trainer.train(train_data=train_data)


def spm_training_preprocess(task_type, task_export_path, config):
    """ Interface function for spm training and spm model backup. 

        Args:
            task_type: Specify task type, only work on text data related task
            config: Yaml config of given task_type 
    """
    assert task_type not in (
        "SSL"), "`spm_train` work on text data related task."

    # Only work on subword tokenizer
    if config["tokenizer"]["type"] == "subword":
        spm_export_dir = os.path.join(task_export_path, "spm")
        spm_model = config["tokenizer"]["config"]["spm_model"]
        spm_vocab = config["tokenizer"]["config"]["spm_vocab"]
        spm_apply_train = config["tokenizer"]["apply_train"]

        if spm_apply_train:
            assert (
                spm_model is None and spm_vocab is None
            ), "If spm_train applied, please set spm_model and spm_vocab as `None`"
            assert "train_config" in config[
                "tokenizer"], "If spm_train applied, please provide related train_config."
            config["tokenizer"]["train_config"][
                "spm_export_path"] = spm_export_dir

            _spm_training(config)

            # Update tokenizer setting with trained spm.
            config["tokenizer"]["config"]["spm_model"] = os.path.join(
                spm_export_dir, "subword.model")
            config["tokenizer"]["config"]["spm_vocab"] = os.path.join(
                spm_export_dir, "subword.vocab")
        else:
            # Backup pre-trained spm model.
            os.makedirs(spm_export_dir, exist_ok=True)
            shutil.copyfile(
                spm_model,
                os.path.join(spm_export_dir, os.path.basename(spm_model)))
            shutil.copyfile(
                spm_vocab,
                os.path.join(spm_export_dir, os.path.basename(spm_vocab)))

    # Return update train config.
    return config
