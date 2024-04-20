# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.04.20
""" Optimizer and Scheduler """

from typing import Union

import math
import torch
import warnings
import numpy as np

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR


def OptimSetup(config):
    # Optimizer and Scheduler Setup inferface
    if config["optimizer"]["type"] == "Adam":
        optimizer = Adam
    elif config["optimizer"]["type"] == "AdamW":
        optimizer = AdamW
    else:
        raise ValueError("{} optimizer is not supported.".format(
            config["optimizer"]["type"]))

    if config["lr_scheduler"]["type"] == "Warmup":
        lr_scheduler = WarmupLR
    elif config["lr_scheduler"]["type"] == "Cosine_Annealing":
        lr_scheduler = CosineAnnealingLR
    elif config["lr_scheduler"]["type"] == "Cosine_Warmup":
        lr_scheduler = CosineWarmupScheduler
    else:
        raise ValueError("{} lr_scheduler is not supported.".format(
            config["lr_scheduler"]["type"]))
    return optimizer, lr_scheduler


class CosineWarmupScheduler(_LRScheduler):
    """ Cosine Scheduler with WarmUp from offical doc """

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().init_(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class WarmupLR(_LRScheduler):
    """ The WarmupLR scheduler
    
        This scheduler is almost same as NoamLR Scheduler except for following
        difference:
        NoamLR:
            lr = optimizer.lr * model_size ** -0.5
                * min(step ** -0.5, step * warmup_step ** -1.5)
        WarmupLR:
            lr = optimizer.lr * warmup_step ** 0.5
                * min(step ** -0.5, step * warmup_step ** -1.5)
        Note that the maximum lr equals to optimizer.lr in this scheduler.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: Union[int, float] = 25000,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        if self.warmup_steps == 0:
            return [lr * step_num**-0.5 for lr in self.base_lrs]
        else:
            return [
                lr * self.warmup_steps**0.5 *
                min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
                for lr in self.base_lrs
            ]

    def set_step(self, step: int):
        self.last_epoch = step


class WarmupPolicy(_LRScheduler):
    """ Adds warmup kwargs and warmup logic to lr policy.
        All arguments should be passed as kwargs for clarity, 
        
        Args:
            warmup_steps: Number of training steps in warmup stage.
            warmup_ratio: Ratio of warmup steps to total steps.
            max_steps: Total number of steps while training or `None` for
                infinite training.
    """

    def __init__(self,
                 optimizer,
                 *,
                 warmup_steps=None,
                 warmup_ratio=None,
                 max_steps=None,
                 min_lr=0.0,
                 last_epoch=-1):
        assert not (warmup_steps is not None and warmup_ratio is not None
                   ), "Either use particular number of step or ratio"
        assert warmup_ratio is None or max_steps is not None, "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.",
                UserWarning)

        step = self.last_epoch

        if step < self.warmup_steps and self.warmup_steps > 0:
            return self._get_warmup_lr(step)

        if (self.max_steps is not None) and (step > self.max_steps):
            return [self.min_lr for _ in self.base_lrs]

        return self._get_lr(step)

    def _get_warmup_lr(self, step):
        lr_val = (step + 1) / (self.warmup_steps + 1)
        return [initial_lr * lr_val for initial_lr in self.base_lrs]

    def _get_lr(self, step):
        """ Simple const lr policy """
        return self.base_lrs


class WarmupHoldPolicy(WarmupPolicy):
    """ Variant of WarmupPolicy which maintains high learning rate for a 
        defined number of steps. All arguments should be passed as kwargs 
        for clarity. 
        
        Args:
            warmup_steps: Number of training steps in warmup stage.
            warmup_ratio: Ratio of warmup steps to total steps.
            hold_steps: Number of training steps to hold the learning rate after 
                warm up.
            hold_ratio: Ratio of hold steps to total steps.
            max_steps: Total number of steps while training or 'None for
                infinite training
    """

    def __init__(self,
                 optimizer,
                 *,
                 warmup_steps=None,
                 warmup_ratio=None,
                 hold_steps=None,
                 hold_ratio=None,
                 max_steps=None,
                 min_lr=0.0,
                 last_epoch=-1):

        assert not (hold_steps is not None and hold_ratio is not None
                   ), "Either use particular number of step or ratio"
        assert hold_ratio is None or max_steps is not None, "If there is a ratio, there should be a total steps"

        self.min_lr = min_lr
        self._last_warmup_1r = 0.0

        # Necessary to duplicate as class attributes are hidden in inner class
        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        if hold_steps is not None:
            self.hold_steps = hold_steps + self.warmup_steps
        elif hold_ratio is not None:
            self.hold_steps = int(hold_ratio * max_steps) + self.warmup_steps
        else:
            self.hold_steps = 0

        super().__init__(
            optimizer,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            max_steps=max_steps,
            last_epoch=last_epoch,
            min_lr=min_lr,
        )

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.", UserWarning)

        step = self.last_epoch

        # Warmup phase
        if step <= self.warmup_steps and self.warmup_steps > 0:
            return self._get_warmup_lr(step)

        # Hold phase
        if (step >= self.warmup_steps) and (step < self.hold_steps):
            return self.base_lrs

        if step > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        return self._get_lr(step)


class NoamHoldAnnealing(WarmupHoldPolicy):
    """ Implementation of the Noam Hold Annealing policy from the SqueezeFormer paper.
        
        Unlike NoamAnnealing, the peak learning rate can be explicitly set for this scheduler.
        The schedule first performs linear warmup, then holds the peak LR, then decays with some schedule 
        for the remainder of the steps. Therefore the min-lr is still dependent on the hyper parameters selected.
        
        It's schedule is determined by three factors-
        
        Warmup Steps: Initial stage, where linear warmup occurs uptil the peak LR is reached. Unlike NoamAnnealing,
            the peak LR is explicitly stated here instead of a scaling factor.
        
        Hold Steps: Intermediate stage, where the peak LR is maintained for some number of steps. In this region, 
            the high peak LR allows the model to converge faster if training is stable. However the high LR 
            may also cause instability during training. Should usually be a significant fraction of training 
            steps (around 30-40% of the entire training steps).
        
        Decay Steps: Final stage, where the LR rapidly decays with some scaling rate (set by decay rate).
            To attain Noam decay, use 0.5, for Squeezeformer recommended decay, use 1.0. The fast decay after 
            prolonged high LR during hold phase allows for rapid convergence.

        References:
            - [Squeezeformer: An Efficient Transformer for Automatic Speech Recognition] (https://arxiv.org/abs/2206.00888)

        Args:
            optimizer: Pytorch compatible Optimizer object. 
            warmup_steps: Number of training steps in warmup stage.
            warmup_ratio: Ratio of warmup steps to total steps.
            hold_steps: Number of training steps to hold the learning rate after warm up 
            hold_ratio: Ratio of hold steps to total steps
            max_steps: Total number of steps while training or `None` for
                infinite training
            decay_rate: Float value describing the polynomial decay after the hold period. Default value
                of 0.5 corresponds to Noam decay.
            min_lr: Minimum learning rate.
    """

    def __init__(self,
                 optimizer,
                 *,
                 max_steps,
                 decay_rate=0.5,
                 min_lr=0.0,
                 last_epoch=-1,
                 **kwargs):

        self.decay_rate = decay_rate
        super().__init__(optimizer=optimizer,
                         max_steps=max_steps,
                         last_epoch=last_epoch,
                         min_lr=min_lr,
                         **kwargs)

    def _get_lr(self, step):
        if self.warmup_steps is None or self.warmup_steps == 0:
            raise ValueError(
                "Noam scheduler cannot be used without warmup steps")
        
        if self.hold_steps > 0:
            hold_steps = self.hold_steps - self.warmup_steps
        else:
            hold_steps = 0

        new_lrs = [
            self._noam_hold_annealing(
                initial_lr,
                step=step,
                warmup_steps=self.warmup_steps,
                hold_steps=hold_steps,
                decay_rate=self.decay_rate,
                min_lr=self.min_lr,
            ) for initial_lr in self.base_lrs
        ]
        return new_lrs

    def _noam_hold_annealing(initial_lr, step, warmup_steps, hold_steps,
                             decay_rate, min_lr):
        # hold_steps = total number of steps to hold the LR, not the warmup + hold steps.
        T_warmup_decay = max(1, warmup_steps**decay_rate)
        T_hold_decay = max(1, (step - hold_steps)**decay_rate)
        lr = (initial_lr * T_warmup_decay) / T_hold_decay
        lr = max(lr, min_lr)
        return lr
