# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.04.20
""" CTC-arch ASR task impl. """

import copy
import glog
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch.distributed.fsdp.wrap import wrap

from dataset.utils import TokenizerSetup
from dataset.frontend.frontend import KaldiWaveFeature, DummyFrontend
from dataset.dataset import AsrEvalDataset, AsrTrainDataset, asr_collate_fn
from model.layer.global_cmvn import GlobalCmvnLayer
from model.encoder.encoder import Encoder
from model.decoder.decoder import Decoder
from model.loss.loss import Loss
from model.utils import AsrMetric, AsrMetricConfig
from optimizer.optim_setup import OptimSetup


class CtcTask(pl.LightningModule):
    """ Build CTC ASR task from Yaml training config """

    def __init__(self, config) -> None:
        super(CtcTask, self).__init__()

        # Split configs by the yaml
        self._tokenizer_config = config["tokenizer"]
        self._dataset_config = config["dataset"]
        self._encoder_config = config["encoder"]
        self._decoder_config = config["decoder"]
        self._loss_config = config["loss"]
        self._metric_config = config["metric"]
        self._optim_config = config["optim_setup"]

        # Build Vanilla CTC-arch ASR system modules, this will be use as base
        # class for upgraded version of CTC like 2-pass.
        self._tokenizer = TokenizerSetup(self._tokenizer_config)
        self._frontend = self._get_frontend(copy.deepcopy(config["dataset"]))
        self._global_cmvn = GlobalCmvnLayer(
            config=self._dataset_config)  # Register CMVN buffer
        self._encoder = Encoder(self._encoder_config)
        self._decoder = Decoder(self._decoder_config)
        self._loss = Loss(self._loss_config)
        self._metric = AsrMetric(config=AsrMetricConfig(**self._metric_config),
                                 tokenizer=self._tokenizer)

    def _get_frontend(self, config):
        # Get Frontend from config to export frontend compute graph
        if config["feat_type"] == "fbank":
            # Set dither as 0.0 when output frontend
            config["feat_config"]["dither"] = 0.0
            return KaldiWaveFeature(**config["feat_config"])
        elif config["feat_type"] == "pcm":
            return DummyFrontend(**config["feat_config"])
        else:
            raise ValueError(
                "Only 'fbank' and 'pcm' feat type supported currently.")

    def non_streaming_inference(self, feats):
        """ Non-streaming inference, just as interface for model test. """
        feats = self._global_cmvn(feats)
        x = self._encoder.non_streaming_inference(feats)
        x = self._decoder.non_streaming_inference(feats)
        x = F.log_softmax(x, dim=-1)
        return x

    def simu_streaming_inference(self, feats):
        """ Simulated-streaming inference, just as interface for model test. """
        feats = self._global_cmvn(feats)
        x = self._encoder.simu_streaming_inference(feats)
        x = self._decoder.simu_streaming_inference(feats)
        x = F.log_softmax(x, dim=-1)
        return x

    def train_dataloader(self):
        """ Set up train dataloader. """

        dataset = AsrTrainDataset(self._dataset_config, self._tokenizer)
        dataloader = DataLoader(dataset=dataset,
                                shuffle=True,
                                collate_fn=asr_collate_fn,
                                batch_size=self._dataset_config["batch_size"],
                                num_workers=4)
        return dataloader

    def val_dataloader(self):
        """ Set up eval dataloader. """

        dataset = AsrEvalDataset(self._dataset_config, self._tokenizer)
        dataloader = DataLoader(dataset=dataset,
                                collate_fn=asr_collate_fn,
                                batch_size=self._dataset_config["batch_size"],
                                num_workers=4)
        return dataloader

    def training_step(self, batch, batch_idx):
        """ DataAgumentation would be process every training step begins,
            so, off-to-go batch input would be {
                "feat": Tensor Float,
                "feat_length": Tensor Long,
                "label": Tensor Long,
                "label_length": Tensor Long,
        """
        feat = self._global_cmvn(batch["feat"])

        encoder_out, encoder_out_length = self._encoder(feat,
                                                        batch["feat_length"])
        decoder_out, decoder_out_length = self._decoder(encoder_out,
                                                        encoder_out_length)

        # Organize batch as Loss API
        loss_input_batch = {
            "log_probs": decoder_out,
            "inputs_length": decoder_out_length,
            "targets": batch["label"],
            "targets_length": batch["label_length"]
        }

        loss = self._loss(loss_input_batch)

        if batch_idx % 100 == 0:
            glog.info(
                "Train (Epoch: {} / Local_steps: {} / Global_steps: {}) loss: {}"
                .format(self.current_epoch, batch_idx, self.global_step, loss))

        self.log("train_loss", loss, sync_dist=True, prog_bar=True, logger=True)
        return loss.mean()

    def validation_step(self, batch, batch_idx):
        """ DataAgumentation would be exluded every eval step begins. """

        feat = self._global_cmvn(batch["feat"])

        encoder_out, encoder_out_length = self._encoder(feat,
                                                        batch["feat_length"])
        decoder_out, decoder_out_length = self._decoder(encoder_out,
                                                        encoder_out_length)

        # Organize batch as Loss API
        loss_input_batch = {
            "log_probs": decoder_out,
            "inputs_length": decoder_out_length,
            "targets": batch["label"],
            "targets_length": batch["label_length"]
        }

        loss = self._loss(loss_input_batch)
        glog.info("Evaluating......")
        wer = self._metric(F.log_softmax(decoder_out, dim=-1),
                           decoder_out_length, batch["label"])

        if batch_idx % 50 == 0:
            glog.info(
                "Eval (Epoch: {} / Local_steps: {} / Global_steps: {}) loss: {} wer: {}"
                .format(self.current_epoch, batch_idx, self.global_step, loss,
                        wer))
        eval_info = {"val_loss": loss, "wer": wer}
        self.log_dict(eval_info, sync_dist=True, prog_bar=True)

    def configure_sharded_model(self):
        # modules are sharded across processes
        # as soon as they are wrapped with 'wrap'.
        # During the forward/backward passes, weights get synced across processes
        # and de-allocated once computation is complete, saving memory.

        # Wraps the layer in a Fully Sharded Wrapper automatically
        self._global_cmvn = wrap(self._global_cmvn)
        self._encoder = wrap(self._encoder)
        self._decoder = wrap(self._decoder)

    def configure_optimizers(self):
        """ Optimizer configuration """
        Optimizer, LR_Scheduler = OptimSetup(self._optim_config)
        if self._optim_config["seperate_lr"]["apply"]:
            params = [{
                "params": self._encoder.parameters(),
                "name": "encoder_lr",
                "lr": self._optim_config["seperate_lr"]["config"]["encoder_lr"]
            }, {
                "params": self._decoder.parameters(),
                "name": "decoder_lr",
                "lr": self._optim_config["seperate_lr"]["config"]["decoder_lr"]
            }]
        else:
            params = self.parameters()

        optimizer = Optimizer(params,
                              **self._optim_config["optimizer"]["config"])
        lr_scheduler = LR_Scheduler(
            optimizer=optimizer, **self._optim_config["lr_scheduler"]["config"])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                **self._optim_config["lr_scheduler"]["step_config"]
            }
        }
