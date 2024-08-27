# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.04.20
""" Nnlm task impl. """

import copy
import glog
import torch
import torch.distributed as dist
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import wrap

from dataset.utils import TokenizerSetup
from dataset.dataset import LmDataset, lm_collate_fn
from dataset.sampler import DynamicBucketBatchSampler
from model.lm.rnn_lm import RnnLm, RnnLmConfig
from model.functions.masking import make_non_pad_mask
from model.loss.loss import Loss
from model.utils import NnLmMetric, NnLmMetricConfig
from optimizer.optim_setup import OptimSetup


class NnLmTask(pl.LightningModule):
    """ Rnn Lm task, for decoding augmentation of asr task. """

    def __init__(self, config) -> None:
        super(NnLmTask, self).__init__()

        # Split configs by the of yaml
        self._tokenizer_config = config["tokenizer"]
        self._dataset_config = config["dataset"]
        self._nnlm_config = config["nnlm"]
        self._loss_config = config["loss"]
        self._metric_config = config["metric"]
        self._optim_config = config["optim_setup"]

        # Build required training module respectivly
        self._tokenizer = TokenizerSetup(self._tokenizer_config)
        self._nnlm = RnnLm(config=RnnLmConfig(**self._nnlm_config))
        self._loss = Loss(self._loss_config)
        self._metric = NnLmMetric(config=NnLmMetricConfig(
            **self._metric_config))

    def train_dataloader(self):
        """ Set up train dataloader. """

        dataset = LmDataset(
            dataset_json=self._dataset_config["train_data"],
            token_min_filter=self._dataset_config["token_min_filter"],
            token_max_filter=self._dataset_config["token_max_filter"],
            tokenizer=self._tokenizer)
        glog.info("Train dataset total tokens: {}.".format(
            dataset.total_data_amount))

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
                                    collate_fn=lm_collate_fn,
                                    batch_sampler=batch_sampler,
                                    num_workers=4)
        else:
            dataloader = DataLoader(
                dataset=dataset,
                sampler=sampler,
                collate_fn=lm_collate_fn,
                batch_size=self._dataset_config["batch_size"],
                num_workers=4)
        return dataloader

    def val_dataloader(self):
        """ Set up eval dataloader. """

        dataset = LmDataset(
            dataset_json=self._dataset_config["eval_data"],
            token_min_filter=self._dataset_config["token_min_filter"],
            token_max_filter=self._dataset_config["token_max_filter"],
            tokenizer=self._tokenizer)
        glog.info("Eval dataset total tokens: {}.".format(
            dataset.total_data_amount))

        sampler = DistributedSampler(dataset,
                                     num_replicas=dist.get_world_size(),
                                     rank=dist.get_rank(),
                                     shuffle=False,
                                     drop_last=False)
        dataloader = DataLoader(dataset=dataset,
                                sampler=sampler,
                                collate_fn=lm_collate_fn,
                                batch_size=self._dataset_config["batch_size"],
                                num_workers=4)
        return dataloader

    def _generate_nnlm_input(self, tokens: torch.Tensor,
                             tokens_length: torch.Tensor):
        """ Generate Lm teacher-forcing auto-regressive input and label:
            Original tokens: [3, 6, 1, 7, 90]
            -> Input: [3, 6, 1, 7]
               Label: [6, 1, 7, 90]
        """
        input_tokens = tokens[:, :-1]
        label_tokens = tokens[:, 1:]
        tokens_length = tokens_length - 1

        return input_tokens.long(), label_tokens.long(), tokens_length.long()

    def training_step(self, batch, batch_idx):
        """ DataAgumentation would be process every training step begins,
            so, off-to-go batch input would be {
                "text": Tensor Long,
                "text_length": Tensor Long,
        """
        input_tokens, label_tokens, tokens_length = self._generate_nnlm_input(
            batch["text"], batch["text_length"])

        logits, logits_length = self._nnlm(input_tokens, tokens_length)

        loss_input_batch = {
            "logits": logits,
            "ori_labels": label_tokens,
            "mask": logits_length,
        }
        loss = self._loss(loss_input_batch)

        if batch_idx % 100 == 0:
            glog.info(
                "Train (Epoch: {} / Local_steps: {} / Global_steps: {}) loss: {}"
                .format(self.current_epoch, batch_idx, self.global_step, loss))
        train_info = {
            "train_loss": loss,
        }
        self.log_dict(train_info, sync_dist=True, prog_bar=True, logger=True)
        return loss.mean()

    def validation_step(self, batch, batch_idx):

        input_tokens, label_tokens, tokens_length = self._generate_nnlm_input(
            batch["text"], batch["text_length"])

        logits, logits_length = self._nnlm(input_tokens, tokens_length)

        loss_input_batch = {
            "logits": logits,
            "ori_labels": label_tokens,
            "mask": logits_length,
        }
        loss = self._loss(loss_input_batch)
        preds = self._loss.predict(logits)
        accs = self._metric(logits=preds,
                            labels=loss_input_batch["ori_labels"],
                            masked_dim=make_non_pad_mask(logits_length).long())

        if batch_idx % 50 == 0:
            glog.info(
                "Eval (Epoch: {} / Local_steps: {} / Global_steps: {}) loss: {} metric: {}"
                .format(self.current_epoch, batch_idx, self.global_step, loss,
                        accs))
        train_info = {"val_loss": loss, **accs}
        self.log_dict(train_info, sync_dist=True, prog_bar=True, logger=True)

    def configure_sharded_model(self):
        # modules are sharded across processes
        # as soon as they are wrapped with 'wrap'.
        # During the forward/backward passes, weights get synced across processes
        # and de-allocated once computation is complete, saving memory.

        # Wraps the layer in a Fully Sharded Wrapper automatically
        self._nnlm = wrap(self._nnlm)
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
