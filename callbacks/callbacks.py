# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.13
""" Callback funcs supporting Training, including official offered 
    callbacks. All should be loaded from this script.
"""

import glog
import os
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from dataset.dataset import AsrTestDataset


class FrontendExport(pl.Callback):
    """ Callback func to save frontend as torchscript when training start, 
        specific designed for deployment. 
    """

    def __init__(self, save_dir) -> None:
        super(FrontendExport, self).__init__()
        self._save_dir = save_dir

    def on_fit_start(self, trainer: pl.Trainer,
                     pl_module: pl.LightningModule) -> None:
        if trainer.global_rank == 0:
            # Applied only on rank 0 device.
            glog.info("Frontend export specified.")
            dummy_pcm = torch.rand(1, 16000)
            torchscipt_frontend = torch.jit.trace(pl_module._frontend,
                                                  example_inputs=dummy_pcm)
            torchscipt_frontend = torch.jit.script(torchscipt_frontend)
            torchscipt_frontend.save(
                os.path.join(self._save_dir, "frontend.script"))
            glog.info("Exported frontend can be found at {}".format(
                os.path.join(self._save_dir, "frontend.script")))


class ComputeGlobalCmvn(pl.Callback):
    """ Compute Global CMN when training starts for stablization of training """

    def __init__(self, feat_dim, frontend_dir, export_dir) -> None:
        super(ComputeGlobalCmvn, self).__init__()
        self._feat_dim = feat_dim
        self._frontend_dir = frontend_dir
        self._export_dir = export_dir

    def on_fit_start(self, trainer: "pl.Trainer",
                     pl_module: "pl.LightningModule") -> None:
        if trainer.global_rank == 0:
            # Create empty tensor to track global cmvn over dataset
            glog.info(
                "Global CMVN specified, compute through whole training dataset..."
            )

            global_mean = torch.zeros(self._feat_dim)  # (feat_dim)
            global_var = torch.ones(self._feat_dim)  # (feat_dim)
            num_frames = 0

            assert hasattr(pl_module, " _dataset_config")
            assert os.path.exists(
                os.path.join(self._frontend_dir, "frontend.script"))
            frontend = os.path.join(self._frontend_dir, "frontend.script")

            dataset = AsrTestDataset(
                dataset_json=pl_module._dataset_config["train_data"],
                frontend=frontend,
                dur_min_filter=pl_module._dataset_config["dur_min_filter"],
                dur_max_filter=pl_module._dataset_config["dur_max_filter"])
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=1,
                                    num_workers=4)
            for i, batch in enumerate(dataloader):
                feat = batch["feat"].squeeze(0)  # (1, T, D) -> (T, D)
                if (i + 1) % 50000 == 0:
                    glog.info("{} utterance done.".format(i + 1))
                global_mean += torch.sum(feat, dim=0)
                global_var += torch.sum(torch.square(feat), dim=0)
                num_frames += feat.shape[0]

            # Compute Global CMVN, clamp var with 1.0e-20
            global_mean = global_mean / num_frames
            global_var = global_var / num_frames - global_mean**2
            global_var = global_var.clamp(min=1.0e-20)
            global_istd = 1.0 / torch.sqrt(global_var)

            # Save global_mean and global_istd for global_cvmn_layer init
            torch.save(global_mean,
                       os.path.join(self._export_dir, "global_mean.t"))
            torch.save(global_istd,
                       os.path.join(self._export_dir, "global_istd.t"))
            glog.info("Done. Build Global CMVN layer for data normalization.")


class LoadGlobalCmvn(pl.Callback):
    """ Load global mean and istd into GlobalCmvnLayer, this seperate design is 
        because when FSDP strategy applied buffer will not synchronized across 
        all GPU node. Therefore, compute global cmvn only on rank 0, and load
        saved mean istd into all ranks.
    """

    def __init__(self, cmvn_dir) -> None:
        super().__init__()

        self._cmvn_dir = cmvn_dir

    def on_train_start(self, trainer: pl.Trainer,
                       pl_module: pl.LightningModule) -> None:
        # Load saved global mean/istd into GlobalCmvnLayer
        global_mean = torch.load(os.path.join(self._cmvn_dir, "global_mean.t"))
        global_istd = torch.load(os.path.join(self._cmvn_dir, "global_istd.t"))
        pl_module._global_cmvn.state_dict()["global_mean"].copy_(global_mean)
        pl_module._global_cmvn.state_dict()["global_istd"].copy_(global_istd)

        glog.info("CMVN loaded.")
