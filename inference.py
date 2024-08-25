# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.24
""" Inference entrypoint. """

import os
import sys
import torch
import yaml
import gflags
import glob
import glog
import logging
import shutil
import pytorch_lightning as pl

from enum import Enum

from task_factory.ctc_task import CtcInference
from task_factory.rnnt_task import RnntInference, PrunedRnntInference
from task_factory.cif_task import CifInference
from tools.model_average import model_average

FLAGS = gflags.FLAGS

gflags.DEFINE_string("inference_config", "config/debug_yaml",
                     "Yaml configuratuon of inference.")


class InferenceFactory(Enum):
    """ Task Factory, build selected task from config """
    ctc_inference = CtcInference
    rnnt_inference = RnntInference
    ctc_hybrid_rnnt_inference = RnntInference
    pruned_rnnt_inference = PrunedRnntInference
    cif_inference = CifInference


def run_inference():
    # ----- Inference set up and logging initialization ------
    FLAGS(sys.argv)
    with open(FLAGS.inference_config, 'r') as config_yaml:
        infer_config = yaml.load(config_yaml.read(), Loader=yaml.FullLoader)
        train_config = yaml.load(open(infer_config["task"]["train_config"],
                                      'r').read(),
                                 Loader=yaml.FullLoader)

    # Use spm_model stored in task_export_dir instead of original configured
    # in train_config if subword tokenizer is set
    if train_config["tokenizer"]["type"] == "subword":
        spm_path = os.path.join(train_config["task"]["export_path"], "spm")
        spm_model = glob.glob(os.path.join(spm_path, "*.model"))[0]
        spm_vocab = glob.glob(os.path.join(spm_path, "*.vocab"))[0]
        train_config["tokenizer"]["config"]["spm_model"] = spm_model
        train_config["tokenizer"]["config"]["spm_vocab"] = spm_vocab

    # For better performance according to the suggestion given by lightning2.0
    torch.set_float32_matmul_precision("medium")

    # Set up load and export path
    TASK_CONFIG = infer_config["task"]
    TASK_TYPE = TASK_CONFIG["type"]
    INFER_EXPORT_PATH = infer_config["task"]["export_path"]

    # Backup inference config and setup logging
    os.makedirs(INFER_EXPORT_PATH, exist_ok=True)
    handler = logging.FileHandler(
        os.path.join(INFER_EXPORT_PATH, "inference.log"))
    shutil.copyfile(
        FLAGS.inference_config,
        os.path.join(INFER_EXPORT_PATH,
                     os.path.basename(FLAGS.inference_config)))
    glog.init()
    glog.logger.addHandler(handler)
    glog.info(infer_config)

    # Inference Setup
    glog.info("{} inference setting up....".format(TASK_TYPE))
    if infer_config["task"]["chkpt_aver"] == True:
        model_average(os.path.abspath(
            os.path.join(train_config["task"]["export_path"], "checkpoints")),
                      aver_best_k=TASK_CONFIG["aver_best_k"],
                      descending=TASK_CONFIG["descending"])
        CHKPT_PATH = os.path.join(train_config["task"]["export_path"],
                                  "checkpoints", "averaged.chkpt")
    else:
        assert infer_config[
            "chkpt_name"] is not None, "Since chkpt_aver not specified, please provide chkpt name."
        CHKPT_PATH = os.path.join(train_config["task"]["export_path"],
                                  "checkpoints", TASK_CONFIG["chkpt_name"])

    asr_inference = InferenceFactory[TASK_TYPE].value.load_from_checkpoint(
        CHKPT_PATH, train_config=train_config, infer_config=infer_config)
    trainer = pl.Trainer(logger=False, **infer_config["trainer"])
    trainer.test(asr_inference)


if __name__ == "__main__":
    run_inference()
