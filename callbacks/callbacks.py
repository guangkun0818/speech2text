# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.13
""" Callback funcs supporting Training, including official offered 
    callbacks. All should be loaded from this script.
"""

import os
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor


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
