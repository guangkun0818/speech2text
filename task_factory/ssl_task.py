# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.11
""" Self-supervised Learning task impl. """

import abc
import copy
import glog
import torch
import torch.distributed as dist
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import wrap
from typing import List, Dict

from dataset.frontend.frontend import FeatType
from dataset.dataset import SslTrainDataset, SslEvalDataset, ssl_collate_fn
from dataset.sampler import DynamicBucketBatchSampler
from model.layer.global_cmvn import GlobalCmvnLayer
from model.ssl.best_rq import BestRQLayer, BestRQLayerConfig, MaskingStrategyConfig
from model.encoder.encoder import Encoder
from model.decoder.decoder import Decoder  # Using decoder as final logits layer
from model.loss.loss import Loss
from model.utils import SslMetric, SslMetricConfig
from optimizer.optim_setup import OptimSetup


class SslTask(pl.LightningModule):
    """ Self-supervised learning task. """

    def __init__(self, config) -> None:
        super(SslTask, self).__init__()

        # Split configs by the of yaml
        self._dataset_config = config["dataset"]
        self._ssl_layer_config = config["ssl_layer"]
        self._encoder_config = config["encoder"]
        self._logits_layer_config = config["logits_layer"]
        self._loss_config = config["loss"]
        assert self._loss_config["loss_select"] in (
            "tot_loss", "mask_loss"
        ), "'loss_select' should be chosen from 'tot_loss', 'mask_loss'"
        self._metric_config = config["metric"]
        self._optim_config = config["optim_setup"]

        # Build required training module respectivly
        self._frontend = self._get_frontend(copy.deepcopy(config["dataset"]))
        self._global_cmvn = GlobalCmvnLayer(
            config=self._dataset_config)  # Register CMVN buffer
        self._ssl_layer = BestRQLayer(
            layer_config=BestRQLayerConfig(
                **self._ssl_layer_config["layer_config"]),
            masking_config=MaskingStrategyConfig(
                **self._ssl_layer_config["masking_config"]))
        self._encoder = Encoder(self._encoder_config)
        self._logits_layer = Decoder(self._logits_layer_config)
        self._loss = Loss(self._loss_config)
        self._metric = SslMetric(config=SslMetricConfig(**self._metric_config))

    def _get_frontend(self, config):
        # Get Frontend from config to export frontend compute graph
        if config["feat_type"] == "fbank":
            # Set dither as 0.0 when output frontend
            config["feat_config"]["dither"] = 0.0

        return FeatType[config["feat_type"]].value(**config["feat_config"])

    def train_dataloader(self):
        """ Set up train dataloader. """

        dataset = SslTrainDataset(self._dataset_config)
        sampler = DistributedSampler(dataset,
                                     num_replicas=dist.get_world_size(),
                                     rank=dist.get_rank(),
                                     shuffle=True,
                                     drop_last=False)
        if self._dataset_config["use_bucket_sampler"]:
            batch_sampler = DynamicBucketBatchSampler(
                sampler=sampler,
                dataset=dataset,
                **self._dataset_config["bucket_sampler_config"])
            dataloader = DataLoader(dataset=dataset,
                                    collate_fn=ssl_collate_fn,
                                    batch_sampler=batch_sampler,
                                    num_workers=4)
        else:
            dataloader = DataLoader(
                dataset=dataset,
                sampler=sampler,
                collate_fn=ssl_collate_fn,
                batch_size=self._dataset_config["batch_size"],
                num_workers=4)
        return dataloader

    def val_dataloader(self):
        """ Set up eval dataloader. """

        dataset = SslEvalDataset(self._dataset_config)
        sampler = DistributedSampler(dataset,
                                     num_replicas=dist.get_world_size(),
                                     rank=dist.get_rank(),
                                     shuffle=False,
                                     drop_last=False)
        dataloader = DataLoader(dataset=dataset,
                                sampler=sampler,
                                collate_fn=ssl_collate_fn,
                                batch_size=self._dataset_config["batch_size"],
                                num_workers=4)
        return dataloader

    def training_step(self, batch, batch_idx):
        """ Training step: Feats -> SslLayer -> Encoder -> LogitsLayer -> Loss 
            batch: {
                "raw_feat": Tensor Float (B, T, D)
                "auged_feat": Tensor Float (B, T, D)
                "feat_length": Tensor Long (B)
        """
        raw_feats = self._global_cmvn(batch["raw_feat"])
        auged_feats = self._global_cmvn(batch["auged_feat"])

        # Output: {"masked_feats", "labels", "masked_dim"}
        output = self._ssl_layer(raw_feats, auged_feats, batch["feat_length"])
        enc_out, enc_out_length = self._encoder(output["masked_feats"],
                                                batch["feat_length"])

        # Logits layer forward
        logits, logits_length = self._logits_layer(enc_out, enc_out_length)

        # Tracking masking proportion on Tensorboard
        stat_info = {
            "mask_rate": output["masked_dim"].sum() / logits_length.sum()
        }
        self.log_dict(stat_info, sync_dist=True, prog_bar=True, logger=True)

        mask_losses = []
        tot_losses = []
        for codebook_id in range(self._ssl_layer.num_codebooks):
            # Organize batch as Loss API
            loss_input_batch = {
                "logits": logits,
                "ori_labels": output["labels"]
                              [codebook_id, :, :],  # Select each codebook
                "mask": output["masked_dim"],
            }
            mask_loss = self._loss(loss_input_batch)  # Loss on Masked frames
            mask_losses.append(mask_loss)

            loss_input_batch = {
                "logits": logits,
                "ori_labels": output["labels"]
                              [codebook_id, :, :],  # Select each codebook
                "mask": logits_length,
            }
            tot_loss = self._loss(loss_input_batch)  # Loss on total sequence
            tot_losses.append(tot_loss)

        # Average over multi-loss
        mask_loss = sum(mask_losses) / self._ssl_layer.num_codebooks
        tot_loss = sum(tot_losses) / self._ssl_layer.num_codebooks

        if self._loss_config["loss_select"] == "tot_loss":
            loss = tot_loss
        elif self._loss_config["loss_select"] == "mask_loss":
            loss = mask_loss

        if batch_idx % 100 == 0:
            glog.info(
                "Train (Epoch: {} / Local_steps: {} / Global_steps: {}) loss: {} tot_loss: {} mask_loss: {}"
                .format(self.current_epoch, batch_idx, self.global_step, loss,
                        tot_loss, mask_loss))
        train_info = {
            "train_loss": loss,
            "train_loss/tot_loss": tot_loss,
            "train_loss/mask_loss": mask_loss
        }
        self.log_dict(train_info, sync_dist=True, prog_bar=True, logger=True)
        return loss.mean()

    def validation_step(self, batch, batch_idx):
        """ Validation step: Feats -> SslLayer -> Encoder -> LogitsLayer -> Loss
                                                                  ↓
                                                              SslMetric
            batch: {
                "raw_feat": Tensor Float (B, T, D)
                "auged_feat": Tensor Float (B, T, D)
                "feat_length": Tensor Long (B)
        """
        raw_feats = self._global_cmvn(batch["raw_feat"])

        # Output: {"masked_feats", "labels", "masked_dim"}
        output = self._ssl_layer(
            raw_feats, raw_feats, batch["feat_length"]
        )  # Using raw_feat as auged_feats in eval stage.
        enc_out, enc_out_length = self._encoder(output["masked_feats"],
                                                batch["feat_length"])

        # Logits layer forward
        logits, logits_length = self._logits_layer(enc_out, enc_out_length)

        mask_losses = []
        tot_losses = []
        all_accs = []
        for codebook_id in range(self._ssl_layer.num_codebooks):
            # Organize batch as Loss API
            loss_input_batch = {
                "logits": logits,
                "ori_labels": output["labels"]
                              [codebook_id, :, :],  # Select each codebook
                "mask": output["masked_dim"],
            }
            mask_loss = self._loss(loss_input_batch)  # Loss on Masked frames
            mask_losses.append(mask_loss)

            loss_input_batch = {
                "logits": logits,
                "ori_labels": output["labels"]
                              [codebook_id, :, :],  # Select each codebook
                "mask": logits_length,
            }
            tot_loss = self._loss(loss_input_batch)  # Loss on total sequence
            tot_losses.append(tot_loss)

            glog.info("Evaluating on codebook {}......".format(codebook_id))
            preds = self._loss.predict(logits)
            accs = self._metric(logits=preds,
                                labels=loss_input_batch["ori_labels"],
                                masked_dim=output["masked_dim"])
            all_accs.append(accs)

        # Average over multi-loss
        mask_loss = sum(mask_losses) / self._ssl_layer.num_codebooks
        tot_loss = sum(tot_losses) / self._ssl_layer.num_codebooks

        accs = self._normalize_metrics(all_accs)

        if self._loss_config["loss_select"] == "tot_loss":
            loss = tot_loss
        elif self._loss_config["loss_select"] == "mask_loss":
            loss = mask_loss

        if batch_idx % 50 == 0:
            glog.info(
                "Eval (Epoch: {} / Local_steps: {} / Global_steps: {}) loss: {} tot_loss: {} mask_loss: {} metric: {}"
                .format(self.current_epoch, batch_idx, self.global_step, loss,
                        tot_loss, mask_loss, accs))
        train_info = {
            "val_loss": loss,
            "val_loss/tot_loss": tot_loss,
            "val_loss/mask_loss": mask_loss,
            **accs
        }
        self.log_dict(train_info, sync_dist=True, prog_bar=True, logger=True)

    def _normalize_metrics(self,
                           metrics: List[Dict[str, float]]) -> Dict[str, float]:

        # Transfer list of accs into final accs;
        final_accs = {}
        for k in metrics[0]:
            final_accs[k] = []

        for m in metrics:
            for k in m:
                final_accs[k].append(m[k])

        for k in final_accs:
            final_accs[k] = sum(final_accs[k]) / len(metrics)

        return final_accs

    def configure_sharded_model(self):
        # modules are sharded across processes
        # as soon as they are wrapped with 'wrap'.
        # During the forward/backward passes, weights get synced across processes
        # and de-allocated once computation is complete, saving memory.

        # Wraps the layer in a Fully Sharded Wrapper automatically
        self._global_cmvn = wrap(self._global_cmvn)
        self._ssl_layer = wrap(self._ssl_layer)
        self._encoder = wrap(self._encoder)
        self._logits_layer = wrap(self._logits_layer)
        self._loss = wrap(self._loss)

    def configure_optimizers(self):
        """ Optimizer configuration """
        Optimizer, LR_Scheduler = OptimSetup(self._optim_config)
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
