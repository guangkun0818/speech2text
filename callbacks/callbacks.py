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
            # Use dummy pcm for torch.jit.trace(frontend)
            # NOTE: why 41360? please refer to dataset/frontend/frontend_test.py

            dummy_pcm = torch.rand(1, 41360)
            torchscipt_frontend = torch.jit.trace(pl_module._frontend,
                                                  example_inputs=dummy_pcm)
            torchscipt_frontend = torch.jit.script(torchscipt_frontend)
            torchscipt_frontend.save(
                os.path.join(self._save_dir, "frontend.script"))


class GlobalCmvn(pl.Callback):
    """ Compute Global CMN when training starts for stablization of training """

    def __init__(self, feat_dim, frontend_dir) -> None:
        super(GlobalCmvn, self).__init__()
        self._feat_dim = feat_dim
        self._frontend_dir = frontend_dir

    def on_fit_start(self, trainer: "pl.Trainer",
                     pl_module: "pl.LightningModule") -> None:
        if trainer.global_rank == 0:
            # Create empty tensor to track global cmv over dataset
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
                frontend=frontend)
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=1,
                                    num_workers=64)
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

            # Update global mean and global_istd of GlobalCmnLayer
            pl_module._global_cmvn.global_mean = global_mean.to(
                pl_module.device)
            pl_module._global_cmvn.global_istd = global_istd.to(
                pl_module.device)
            glog.info("Done. Build Global CMVN layer for data normalization.")
