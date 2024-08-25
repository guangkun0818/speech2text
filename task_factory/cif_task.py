# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.15
""" CIF ASR task impl. """

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

from dataset.utils import TokenizerSetup
from dataset.frontend.frontend import FeatType
from dataset.dataset import AsrEvalDataset, AsrTrainDataset, asr_collate_fn
from dataset.sampler import DynamicBucketBatchSampler
from model.layer.global_cmvn import GlobalCmvnLayer
from model.encoder.encoder import Encoder
from model.cif.cif_layer import CifLayer, CifLayerConfig
from model.decoder.decoder import Decoder
from model.loss.loss import Loss
from model.utils import AsrMetric, AsrMetricConfig
from optimizer.optim_setup import OptimSetup

from task_factory.asr_inference import AbcAsrInference
from model.decoding import DecodingFactory, batch_search


class BaseCifTask(pl.LightningModule):
    """ Base CIF task impl """

    def __init__(self, config) -> None:
        super(BaseCifTask, self).__init__()

        # Split configs by the yaml
        self._tokenizer_config = config["tokenizer"]
        self._dataset_config = config["dataset"]
        self._encoder_config = config["encoder"]
        self._cif_layer_config = config["cif_layer"]
        self._decoder_config = config["decoder"]
        self._loss_config = config["loss"]
        self._metric_config = config["metric"]
        self._optim_config = config["optim_setup"]

        # Modules fo rnnt task building.
        self._tokenizer = TokenizerSetup(self._tokenizer_config)
        self._frontend = self._get_frontend(copy.deepcopy(config["dataset"]))
        self._global_cmvn = GlobalCmvnLayer(
            config=self._dataset_config)  # Register CMVN buffer
        self._encoder = Encoder(self._encoder_config)
        self._cif_layer = CifLayer(config=CifLayerConfig(
            **self._cif_layer_config))
        self._decoder = Decoder(self._decoder_config)
        self._mae_loss = Loss(self._loss_config["mae_loss"])
        self._aed_loss = Loss(self._loss_config["aed_loss"])
        self._metric = AsrMetric(config=AsrMetricConfig(**self._metric_config),
                                 tokenizer=self._tokenizer)

    def _get_frontend(self, config):
        # Get Frontend from config to export frontend compute graph
        if config["feat_type"] == "fbank":
            # Set dither as 0.0 when output frontend
            config["feat_config"]["dither"] = 0.0

        return FeatType[config["feat_type"]].value(**config["feat_config"])

    def non_streaming_inference(self, feats):
        ...

    def simu_streaming_inference(self, feats):
        ...

    def train_dataloader(self):
        """ Set up train dataloader. """

        dataset = AsrTrainDataset(self._dataset_config, self._tokenizer)
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
                                    collate_fn=asr_collate_fn,
                                    batch_sampler=batch_sampler,
                                    num_workers=4)
        else:
            dataloader = DataLoader(
                dataset=dataset,
                sampler=sampler,
                collate_fn=asr_collate_fn,
                batch_size=self._dataset_config["batch_size"],
                num_workers=4)
        return dataloader

    def val_dataloader(self):
        """ Set up eval dataloader. """

        dataset = AsrEvalDataset(self._dataset_config, self._tokenizer)
        sampler = DistributedSampler(dataset,
                                     num_replicas=dist.get_world_size(),
                                     rank=dist.get_rank(),
                                     shuffle=False,
                                     drop_last=False)
        dataloader = DataLoader(dataset=dataset,
                                sampler=sampler,
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
        self._cif_layer = wrap(self._cif_layer)
        self._decoder = wrap(self._decoder)

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


class CifTask(BaseCifTask):
    """ Vanilla CIF task impl, Non-causal """

    def __init__(self, config) -> None:
        super(CifTask, self).__init__(config=config)

        self._mae_loss_weight = self._loss_config["mae_loss_weight"]

    def training_step(self, batch, batch_idx):
        """ DataAgumentation would be process every training step begins,
            so, off-to-go batch input would be {
                "feat": Tensor Float,
                "feat_length": Tensor Long,
                "label": Tensor Long,
                "label_length": Tensor Long,
        """
        feat = self._global_cmvn(batch["feat"])

        # Encoder forward
        encoder_out, encoder_out_length = self._encoder(feat,
                                                        batch["feat_length"])
        # CIF layer forward
        acoustic_embeds, cif_peak, token_num_hat, alphas = self._cif_layer(
            encoder_out, encoder_out_length, batch["label"],
            batch["label_length"])
        # Decoder forward
        decoder_out, decoder_out_length = self._decoder(acoustic_embeds,
                                                        batch["label_length"])

        # Organize batch as Loss API
        mae_loss_input_batch = {
            "tokens_length": batch["label_length"],
            "pre_tokens_length": token_num_hat,
        }
        mae_loss = self._mae_loss_weight * self._mae_loss(mae_loss_input_batch)

        aed_loss_input_batch = {
            "logits": decoder_out,
            "ori_labels": batch["label"],
            "mask": batch["label_length"],
        }
        aed_loss = self._aed_loss(aed_loss_input_batch)

        loss = mae_loss + aed_loss

        if batch_idx % 100 == 0:
            glog.info(
                "Train (Epoch: {} / Local_steps: {} / Global_steps: {}) loss: {} aed_loss {} mae_loss: {}"
                .format(self.current_epoch, batch_idx, self.global_step, loss,
                        aed_loss, mae_loss))
        train_info = {
            "train_loss": loss,
            "train_loss/aed_loss": aed_loss,
            "train_loss/mae_loss": mae_loss,
        }
        self.log_dict(train_info, sync_dist=True, prog_bar=True, logger=True)

        return loss.mean()

    def validation_step(self, batch, batch_idx):
        """ DataAgumentation would be exluded every eval step begins. """

        feat = self._global_cmvn(batch["feat"])

        # Encoder forward
        encoder_out, encoder_out_length = self._encoder(feat,
                                                        batch["feat_length"])
        # CIF layer forward for loss
        acoustic_embeds, cif_peak, token_num_hat, alphas = self._cif_layer(
            encoder_out, encoder_out_length, batch["label"],
            batch["label_length"])
        # Decoder forward
        decoder_out, decoder_out_length = self._decoder(acoustic_embeds,
                                                        batch["label_length"])

        # Organize batch as Loss API
        mae_loss_input_batch = {
            "tokens_length": batch["label_length"],
            "pre_tokens_length": token_num_hat,
        }
        mae_loss = self._mae_loss_weight * self._mae_loss(mae_loss_input_batch)

        aed_loss_input_batch = {
            "logits": decoder_out,
            "ori_labels": batch["label"],
            "mask": batch["label_length"],
        }
        aed_loss = self._aed_loss(aed_loss_input_batch)

        loss = mae_loss + aed_loss

        # CIF/Decoder layer inference for metric, num of token will be predicted through
        # this stage.
        acoustic_embeds, cif_peak, token_num_hat, alphas = self._cif_layer(
            encoder_out, encoder_out_length)
        decoder_out, decoder_out_length = self._decoder(acoustic_embeds,
                                                        token_num_hat.long())

        glog.info("Evaluating......")
        preds = self._aed_loss.predict(decoder_out)
        wer = self._metric(preds, decoder_out_length, batch["label"])

        if batch_idx % 100 == 0:
            glog.info(
                "Eval (Epoch: {} / Local_steps: {} / Global_steps: {}) loss: {} aed_loss {} mae_loss: {} wer: {}"
                .format(self.current_epoch, batch_idx, self.global_step, loss,
                        aed_loss, mae_loss, wer))
        train_info = {
            "val_loss": loss,
            "val_loss/aed_loss": aed_loss,
            "val_loss/mae_loss": mae_loss,
            "wer": wer
        }
        self.log_dict(train_info, sync_dist=True, prog_bar=True, logger=True)


class CifInference(AbcAsrInference, CifTask):
    """ CIF task inference """

    def __init__(self, infer_config, train_config) -> None:
        # Init wil mro
        super(CifInference, self).__init__(infer_config=infer_config)
        super(AbcAsrInference, self).__init__(config=train_config)
        self._decoding_sess = DecodingFactory[
            self._decoding_config["type"]].value(
                tokenizer=self._tokenizer, **self._decoding_config["config"])

    def test_step(self, batch, batch_idx):
        feat = self._global_cmvn(batch["feat"])

        encoder_out, encoder_out_length = self._encoder(feat,
                                                        batch["feat_length"])

        # CIF/Decoder layer inference for metric, num of token will be predicted through
        # this stage.
        acoustic_embeds, cif_peak, token_num_hat, alphas = self._cif_layer(
            encoder_out, encoder_out_length)

        decoder_out, decoder_out_length = self._decoder(acoustic_embeds,
                                                        token_num_hat.long())

        log_probs = self._aed_loss.predict(decoder_out)

        decoded_texts = batch_search(log_probs,
                                     decoder_out_length,
                                     decode_session=self._decoding_sess)

        self._export_decoded_results(batch["audio_filepath"], decoded_texts,
                                     batch["text"])
        self._predicton += decoded_texts
        self._reference += batch["text"]
