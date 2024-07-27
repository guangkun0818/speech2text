# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.04.20
""" Rnnt-arch ASR task impl. """

import abc
import copy
import glog
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
from model.predictor.predictor import Predictor
from model.joiner.joiner import Joiner, JoinerConfig
from model.loss.loss import Loss
from model.utils import AsrMetric, AsrMetricConfig
from optimizer.optim_setup import OptimSetup


class BaseRnntTask(pl.LightningModule):
    """ Base Rnnt task setting, custom-designed Rnnt task should inherited 
        from this abstract task.
    """

    def __init__(self, config) -> None:
        super(BaseRnntTask, self).__init__()

        # Split configs by the yaml
        self._tokenizer_config = config["tokenizer"]
        self._dataset_config = config["dataset"]
        self._encoder_config = config["encoder"]
        self._decoder_config = config["decoder"]
        self._predictor_config = config["predictor"]
        self._joiner_config = config["joiner"]
        self._metric_config = config["metric"]
        self._optim_config = config["optim_setup"]

        # Modules fo rnnt task building.
        self._tokenizer = TokenizerSetup(self._tokenizer_config)
        self._frontend = self._get_frontend(copy.deepcopy(config["dataset"]))
        self._global_cmvn = GlobalCmvnLayer(
            config=self._dataset_config)  # Register CMVN buffer
        self._encoder = Encoder(self._encoder_config)
        self._decoder = Decoder(self._decoder_config)
        self._predictor = Predictor(self._predictor_config)
        self._joiner = Joiner(config=JoinerConfig(**self._joiner_config))
        self._metric = AsrMetric(config=AsrMetricConfig(**self._metric_config),
                                 tokenizer=self._tokenizer,
                                 predictor=self._predictor,
                                 joiner=self._joiner)

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
        ...

    def simu_streaming_inference(self, feats):
        ...

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

    @abc.abstractmethod
    def training_step(self, batch, batch_idx):
        # Different training step should be implemented corresponding to given
        # configured task.
        ...

    @abc.abstractmethod
    def validation_step(self, batch, batch_idx):
        # Different validation step should be implemented corresponding to given
        # configured task.
        ...

    def configure_sharded_model(self):
        # modules are sharded across processes
        # as soon as they are wrapped with 'wrap'.
        # During the forward/backward passes, weights get synced across processes
        # and de-allocated once computation is complete, saving memory.

        # Wraps the layer in a Fully Sharded Wrapper automatically
        self._global_cmvn = wrap(self._global_cmvn)
        self._encoder = wrap(self._encoder)
        self._decoder = wrap(self._decoder)
        self._predictor = wrap(self._predictor)
        self._joiner = wrap(self._joiner)

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
            }, {
                "params":
                    self._predictor.parameters(),
                "name":
                    "predictor_lr",
                "lr":
                    self._optim_config["seperate_lr"]["config"]["predictor_lr"]
            }, {
                "params": self._joiner.parameters(),
                "name": "joiner_lr",
                "lr": self._optim_config["seperate_lr"]["config"]["joiner_lr"]
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


class RnntTask(BaseRnntTask):
    """ Vanilla Rnnt Task, unpruned rnnt loss. """

    def __init__(self, config) -> None:
        super(RnntTask, self).__init__(config)
        self._loss_config = config["loss"]
        self._loss = Loss(self._loss_config)

    def training_step(self, batch, batch_idx):
        """ DataAgumentation would be process every training step begins,
            so, off-to-go batch input would be {
                "feat": Tensor Float,
                "feat_length": Tensor Long,
                "label": Tensor Long,
                "label_length": Tensor Long,
        """
        feat = self._global_cmvn(batch["feat"])

        # Encoder foward
        encoder_out, encoder_out_length = self._encoder(feat,
                                                        batch["feat_length"])
        # Decoder forward
        decoder_out, decoder_out_length = self._decoder(encoder_out,
                                                        encoder_out_length)
        # Predictor forward
        predictor_state = self._predictor.init_state()
        predictor_out, predictor_length, _ = self._predictor(
            batch["label"], batch["label_length"], predictor_state)
        # Joiner forward
        joiner_out, boundary, ranges, _ = self._joiner(decoder_out,
                                                       decoder_out_length,
                                                       predictor_out,
                                                       predictor_length,
                                                       batch["label"])

        # Organize batch as Loss API
        loss_input_batch = {
            "log_probs": joiner_out,
            "inputs_length": decoder_out_length,
            "targets": batch["label"],
            "targets_length": batch["label_length"],
            "boundary": boundary,
            "ranges": ranges  # API compliance with pruned rnnt.
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

        # Encoder foward
        encoder_out, encoder_out_length = self._encoder(feat,
                                                        batch["feat_length"])
        # Decoder forward
        decoder_out, decoder_out_length = self._decoder(encoder_out,
                                                        encoder_out_length)
        # Predictor forward
        predictor_state = self._predictor.init_state()
        predictor_out, predictor_length, _ = self._predictor(
            batch["label"], batch["label_length"], predictor_state)
        # Joiner forward
        joiner_out, boundary, ranges, _ = self._joiner(decoder_out,
                                                       decoder_out_length,
                                                       predictor_out,
                                                       predictor_length,
                                                       batch["label"])

        # Organize batch as Loss API
        loss_input_batch = {
            "log_probs": joiner_out,
            "inputs_length": decoder_out_length,
            "targets": batch["label"],
            "targets_length": batch["label_length"],
            "boundary": boundary,
            "ranges": ranges  # API compliance with pruned rnnt.
        }

        loss = self._loss(loss_input_batch)
        glog.info("Evaluating......")
        wer = self._metric(decoder_out, decoder_out_length, batch["label"])

        if batch_idx % 50 == 0:
            glog.info(
                "Eval (Epoch: {} / Local_steps: {} / Global_steps: {}) loss: {}"
                .format(self.current_epoch, batch_idx, self.global_step, loss))

        eval_info = {"val_loss": loss, "wer": wer}
        self.log_dict(eval_info, sync_dist=True, prog_bar=True)


class PrunedRnntTask(BaseRnntTask):
    """ k2 Pruned Rnnt task. """

    def __init__(self, config) -> None:
        super(PrunedRnntTask, self).__init__(config=config)

        assert config["loss"]["model"] == "Pruned_Rnnt"
        self._loss_config = config["loss"]
        self._simple_loss_scale = config["loss"]["simple_loss_scale"]
        self._pruned_loss_scale = config["loss"]["pruned_loss_scale"]
        self._loss = Loss(self._loss_config)
        self._enable_ctc = self._loss_config["enable_ctc"]

        if self._enable_ctc:
            self._ctc_config = {
                "model": "CTC",
                "config": {
                    **self._loss_config["ctc_config"]
                }
            }
            self._ctc_loss = Loss(self._ctc_config)
            self._ctc_projector_config = config["ctc_projector"]
            self._ctc_projector = Decoder(self._ctc_projector_config)

    def training_step(self, batch, batch_idx):
        """ DataAgumentation would be process every training step begins,
            so, off-to-go batch input would be {
                "feat": Tensor Float,
                "feat_length": Tensor Long,
                "label": Tensor Long,
                "label_length": Tensor Long,
        """
        feat = self._global_cmvn(batch["feat"])

        # Encoder foward
        encoder_out, encoder_out_length = self._encoder(feat,
                                                        batch["feat_length"])
        # Decoder forward
        decoder_out, decoder_out_length = self._decoder(encoder_out,
                                                        encoder_out_length)
        # Predictor forward
        predictor_state = self._predictor.init_state()
        predictor_out, predictor_length, _ = self._predictor(
            batch["label"], batch["label_length"], predictor_state)

        # Joiner forward
        joiner_out, boundary, ranges, simple_loss = self._joiner(
            decoder_out, decoder_out_length, predictor_out, predictor_length,
            batch["label"])

        # Organize batch as Loss API
        loss_input_batch = {
            "log_probs": joiner_out,
            "inputs_length": decoder_out_length,
            "targets": batch["label"],
            "targets_length": batch["label_length"],
            "boundary": boundary,
            "ranges": ranges  # API compliance with pruned rnnt.
        }

        pruned_loss = self._loss(loss_input_batch)

        if self._enable_ctc:
            logits, logits_length = self._ctc_projector(decoder_out,
                                                        decoder_out_length)
            ctc_loss_input_batch = {
                "log_probs": logits,
                "inputs_length": logits_length,
                "targets": batch["label"],
                "targets_length": batch["label_length"],
            }

            ctc_loss = self._ctc_loss(ctc_loss_input_batch)
            loss = self._simple_loss_scale * simple_loss + self._pruned_loss_scale * pruned_loss + ctc_loss
        else:
            ctc_loss = 0.0
            loss = self._simple_loss_scale * simple_loss + self._pruned_loss_scale * pruned_loss

        if batch_idx % 100 == 0:
            glog.info(
                "Train (Epoch: {} / Local_steps: {} / Global_steps: {}) loss: {} simple_loss {} pruned_loss: {} ctc_loss {}"
                .format(self.current_epoch, batch_idx, self.global_step, loss,
                        simple_loss, pruned_loss, ctc_loss))
        train_info = {
            "train_loss": loss,
            "train_loss/simple_loss": simple_loss,
            "train_loss/pruned_loss": pruned_loss,
            "train_loss/ctc_loss": ctc_loss,
        }
        self.log_dict(train_info, sync_dist=True, prog_bar=True, logger=True)

        return loss.mean()

    def validation_step(self, batch, batch_idx):
        """ DataAgumentation would be exluded every eval step begins. """

        feat = self._global_cmvn(batch["feat"])

        # Encoder foward
        encoder_out, encoder_out_length = self._encoder(feat,
                                                        batch["feat_length"])
        # Decoder forward
        decoder_out, decoder_out_length = self._decoder(encoder_out,
                                                        encoder_out_length)
        # Predictor forward
        predictor_state = self._predictor.init_state()
        predictor_out, predictor_length, _ = self._predictor(
            batch["label"], batch["label_length"], predictor_state)

        # Joiner forward
        joiner_out, boundary, ranges, simple_loss = self._joiner(
            decoder_out, decoder_out_length, predictor_out, predictor_length,
            batch["label"])

        # Organize batch as Loss API
        loss_input_batch = {
            "log_probs": joiner_out,
            "inputs_length": decoder_out_length,
            "targets": batch["label"],
            "targets_length": batch["label_length"],
            "boundary": boundary,
            "ranges": ranges  # API compliance with pruned rnnt.
        }

        pruned_loss = self._loss(loss_input_batch)

        if self._enable_ctc:
            logits, logits_length = self._ctc_projector(decoder_out,
                                                        decoder_out_length)
            ctc_loss_input_batch = {
                "log_probs": logits,
                "inputs_length": logits_length,
                "targets": batch["label"],
                "targets_length": batch["label_length"],
            }

            ctc_loss = self._ctc_loss(ctc_loss_input_batch)
            loss = self._simple_loss_scale * simple_loss + self._pruned_loss_scale * pruned_loss + ctc_loss
        else:
            ctc_loss = 0.0
            loss = self._simple_loss_scale * simple_loss + self._pruned_loss_scale * pruned_loss

        glog.info("Evaluating......")
        wer = self._metric(decoder_out, decoder_out_length, batch["label"])

        if batch_idx % 50 == 0:
            glog.info(
                "Eval (Epoch: {} / Local_steps: {} / Global_steps: {}) loss: {} simple_loss {} pruned_loss: {} ctc_loss: {} wer: {}"
                .format(self.current_epoch, batch_idx, self.global_step, loss,
                        simple_loss, pruned_loss, ctc_loss, wer))
        eval_info = {
            "val_loss": loss,
            "val_loss/simple_loss": simple_loss,
            "val_loss/pruned_loss": pruned_loss,
            "val_loss/ctc_loss": ctc_loss,
            "wer": wer
        }
        self.log_dict(eval_info, sync_dist=True, prog_bar=True, logger=True)

    def configure_sharded_model(self):
        # modules are sharded across processes
        # as soon as they are wrapped with 'wrap'.
        # During the forward/backward passes, weights get synced across processes
        # and de-allocated once computation is complete, saving memory.

        # Wraps the layer in a Fully Sharded Wrapper automatically
        self._global_cmvn = wrap(self._global_cmvn)
        self._encoder = wrap(self._encoder)
        self._decoder = wrap(self._decoder)
        self._predictor = wrap(self._predictor)
        self._joiner = wrap(self._joiner)
        if self._enable_ctc:
            self._ctc_projector = wrap(self._ctc_projector)

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
            }, {
                "params":
                    self._predictor.parameters(),
                "name":
                    "predictor_lr",
                "lr":
                    self._optim_config["seperate_lr"]["config"]["predictor_lr"]
            }, {
                "params": self._joiner.parameters(),
                "name": "joiner_lr",
                "lr": self._optim_config["seperate_lr"]["config"]["joiner_lr"]
            }]
            if self._enable_ctc:
                params.append({
                    "params":
                        self._ctc_projector.parameters(),
                    "name":
                        "ctc_projector_lr",
                    "lr":
                        self._optim_config["seperate_lr"]["config"]
                        ["ctc_projector_lr"]
                })
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
