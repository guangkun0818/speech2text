# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.04.20
""" Task entrypoint, build your task with given config """

import os
import sys
import torch
import yaml
import gflags
import glog
import logging
import random
import shutil
import pytorch_lightning as pl

from enum import Enum, unique
from pytorch_lightning import loggers as pl_loggers

import callbacks.callbacks as callbacks
from tools.model_average import model_average
from tools.spm_train import spm_training_preprocess

from task_factory.cif_task import CifTask
from task_factory.ctc_task import CtcTask
from task_factory.rnnt_task import RnntTask, CtcHybridRnnt, PrunedRnntTask
from task_factory.ssl_task import SslTask
from task_factory.nnlm_task import NnLmTask

FLAGS = gflags.FLAGS

gflags.DEFINE_string("training_config", "config/debug_yaml",
                     "Yaml configuratuon of task.")


@unique
class TaskFactory(Enum):
    """ Task Factory, build selected task from config """
    CTC = CtcTask
    Rnnt = RnntTask
    CTC_Hybrid_Rnnt = CtcHybridRnnt
    Pruned_Rnnt = PrunedRnntTask
    SSL = SslTask
    CIF = CifTask
    NNLM = NnLmTask


def run_task():
    torch.manual_seed(1234)
    random.seed(1234)  # For reproducibility

    # ---- Set up logging initialization. ----
    FLAGS(sys.argv)
    with open(FLAGS.training_config, 'r') as config_yaml:
        config = yaml.load(config_yaml.read(), Loader=yaml.FullLoader)

    TASK_TYPE = config["task"]["type"]
    TASK_NAME = config["task"]["name"]
    TASK_EXPORT_PATH = config["task"]["export_path"]

    os.makedirs(TASK_EXPORT_PATH, exist_ok=True)
    handler = logging.FileHandler(os.path.join(TASK_EXPORT_PATH, "run.log"))
    glog.init()
    glog.logger.addHandler(handler)

    # Back up training config yaml.
    glog.info("{} Task building....".format(TASK_TYPE))
    shutil.copyfile(
        FLAGS.training_config,
        os.path.join(TASK_EXPORT_PATH, os.path.basename(FLAGS.training_config)))

    glog.info(config)

    # ---- Preprocessing. ----
    config = spm_training_preprocess(TASK_TYPE, TASK_EXPORT_PATH, config=config)

    # ---- Set up task. ----
    # For better performance according to suggestion by lightning 2.0
    torch.set_float32_matmul_precision("medium")

    # Setup pretrained-model if finetune/base_model is set
    if config["finetune"]["base_model"]:
        # If base model of finetune is set, then finetune from base model
        if os.path.isfile(config["finetune"]["base_model"]):
            task = TaskFactory[TASK_TYPE].value.load_from_checkpoint(
                config["finetune"]["base_model"], config=config, strict=False)
        elif os.path.isdir(config["finetune"]["base_model"]):
            model_average(config["finetune"]["base_model"])
            pretrain_model = os.path.join(config["finetune"]["base_model"],
                                          "averaged.chkpt")
            task = TaskFactory[TASK_TYPE].value.load_from_checkpoint(
                pretrain_model, config=config, strict=False)
    else:
        task = TaskFactory[TASK_TYPE].value(config)

    # ---- Set up callbacks and tensorboard logs ----
    chkpt_filename = TASK_NAME + "-{epoch}-{val_loss:.2f}" + "-{%s:.2f}" % config[
        "callbacks"]["model_chkpt_config"]["monitor"]
    chkpt_callback = callbacks.ModelCheckpoint(
        dirpath=os.path.join(TASK_EXPORT_PATH, "checkpoints"),
        filename=chkpt_filename,
        **config["callbacks"]
        ["model_chkpt_config"])  # Callbacks save chkpt of model.

    lr_monitor = callbacks.LearningRateMonitor(logging_interval='step')
    tb_logger = pl_loggers.TensorBoardLogger(TASK_EXPORT_PATH)
    callback_funcs = [chkpt_callback, lr_monitor]

    # Export frontend as torchscript for deployment if specified.
    if config["callbacks"]["frontend_save"]:
        frontend_save = callbacks.FrontendExport(save_dir=TASK_EXPORT_PATH)
        callback_funcs.append(frontend_save)

    # Apply GlobalCmvn if specified
    if config["callbacks"]["global_cmvn"]["apply"] and config["resume"] is None:
        # Global CMVN only apply to fbank frontend and when start
        # training from stretch.
        assert "fbank" == config["dataset"]["feat_type"]
        assert "num_mel_bins" in config["dataset"]["feat_config"]
        feat_dim = config["dataset"]["feat_config"]["num_mel_bins"]
        if config["callbacks"]["global_cmvn"]["pre_compute_cmvn"] is None:
            compute_cmvn = callbacks.ComputeGlobalCmvn(
                feat_dim=feat_dim,
                frontend_dir=TASK_EXPORT_PATH,
                export_dir=TASK_EXPORT_PATH)
            callback_funcs.append(compute_cmvn)
        else:
            # If pre_compute_cmvn provide, backup global_mean and global_istd.
            shutil.copyfile(
                os.path.join(
                    config["callbacks"]["global_cmvn"]["pre_compute_cmvn"],
                    "global_mean.t"),
                os.path.join(TASK_EXPORT_PATH, "global_mean.t"))
            shutil.copyfile(
                os.path.join(
                    config["callbacks"]["global_cmvn"]["pre_compute_cmvn"],
                    "global_istd.t"),
                os.path.join(TASK_EXPORT_PATH, "global_istd.t"))
            load_cmvn = callbacks.LoadGlobalCmvn(cmvn_dir=TASK_EXPORT_PATH)
            callback_funcs.append(load_cmvn)

    # ---- Setup trainer from config ----
    trainer = pl.Trainer(**config["trainer"],
                         logger=tb_logger,
                         log_every_n_steps=2,
                         callbacks=callback_funcs)
    # If resume is set, then training will resume from given chkpt
    trainer.fit(task, ckpt_path=config["resume"])


if __name__ == "__main__":
    run_task()
