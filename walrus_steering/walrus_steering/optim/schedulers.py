"""
Copied from PDE Arena
"""

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import math
import warnings
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between
    warmup_start_lr and base_lr followed by a cosine annealing schedule between base_lr and
    eta_min."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_lr_factor: float = 0.1,
        decay_lr_factor: float = 0.1,
        step_mult_factor: float = 1,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr_factor (float): Multiplier of base lr used at start of warmup. Default: 0.1
            decay_lr_factor (float): Multiplier of base lr used at end of decay cycle. Default: 0.1
            step_mult_factor (float): Multiplier for all epoch numbers. Default: 1
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = int(step_mult_factor * warmup_epochs)
        self.max_epochs = int(step_mult_factor * max_epochs)
        self.warmup_lr_factor = warmup_lr_factor
        self.decay_lr_factor = decay_lr_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if self.last_epoch == 0:
            return [self.warmup_lr_factor * base_lr for base_lr in self.base_lrs] * len(
                self.base_lrs
            )
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr * (1 - self.warmup_lr_factor)) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if (self.last_epoch - 1 - self.max_epochs) % (
            2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr * (1 - self.cooldown_factor))
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - base_lr * self.decay_lr_factor)
            + base_lr * self.decay_lr_factor
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs)
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_lr_factor * base_lr
                + self.last_epoch
                * (base_lr * (1 - self.warmup_lr_factor))
                / max(1, self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            base_lr * self.decay_lr_factor
            + 0.5
            * (base_lr * (1 - self.decay_lr_factor))
            * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            for base_lr in self.base_lrs
        ]


class InverseSqrtLinearRamps(_LRScheduler):
    """Sets the learning rate to follow a linear warmup, then an infinite decay phase,
    followed by a linear off-ramp decay to eta_min."""

    def __init__(
        self,
        optimizer: Optimizer,
        max_epochs: int,
        warmup_epochs: int = 1,
        cooldown_epochs: int = 1,
        warmup_lr_factor: float = 0.1,
        cooldown_lr_factor: float = 0.001,
        step_mult_factor: float = 1,
        last_epoch: int = -1,
        offset: float = 128,
        exponent: float = .5,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            max_epochs (int): Maximum number of iterations
            warmup_epochs (int): Number of epochs for linear warm-up.
            cooldown_epochs (int): Number of epochs for linear cooldown.
            warmup_lr_factor (float): Starting learning rate for warm-up phase.
            cooldown_lr_factor (float): Final learning rate for cooldown phase.
            step_mult_factor (float): Multiplier for all epoch numbers. Default: 1.
            last_epoch (int): Index of the last epoch. Default: -1.
        """
        self.warmup_epochs = int(step_mult_factor * warmup_epochs)
        self.cooldown_epochs = int(step_mult_factor * cooldown_epochs)
        self.max_epochs = int(step_mult_factor * max_epochs)
        self.in_infinite_phase = False
        self.steps_before_off_ramp = self.max_epochs - self.cooldown_epochs
        self.warmup_lr_factor = warmup_lr_factor
        self.cooldown_lr_factor = cooldown_lr_factor
        self.offset = offset
        self.exponent = exponent
        super(InverseSqrtLinearRamps, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate for each phase."""
        if self.last_epoch == 0:
            return [self.warmup_lr_factor * base_lr for base_lr in self.base_lrs]
        elif self.last_epoch <= self.warmup_epochs:
            return [
                group["lr"]
                * (
                    1.0
                    + (1. - self.warmup_lr_factor)
                    / (
                        self.warmup_epochs * self.warmup_lr_factor
                        + (self.last_epoch - 1) * (1. - self.warmup_lr_factor)
                    )
                )
                for group in self.optimizer.param_groups
            ]
        elif self.warmup_epochs < self.last_epoch <= self.steps_before_off_ramp:
            # Infinite Phase: Inverse square decay
            step = self.last_epoch - self.warmup_epochs + 1
            return [
                group["lr"] * (1 + (max(1, step - 1) + self.offset) ** self.exponent - self.offset**self.exponent) / (1 + (step + self.offset)**self.exponent - self.offset**self.exponent)
                for group in self.optimizer.param_groups
            ]
        else:
            # Off-ramp Phase: Linear decay from end of infinite phase to cooldown_final_lr
            ramp_epoch = self.last_epoch - self.steps_before_off_ramp - 1
            return [group["lr"] * (
                    1.0 
                    + (self.cooldown_lr_factor - 1.0)
                    / (
                        self.cooldown_epochs 
                        + (ramp_epoch - 1.) * (self.cooldown_lr_factor - 1.) 
                    )
            )
            for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self) -> List[float]:
        """Closed-form solution for the scheduler if epoch is passed to the step function."""
        if self.last_epoch <= self.warmup_epochs:
            return [
                base_lr 
                * (
                    self.warmup_lr_factor
                    + (1 - self.warmup_lr_factor)
                    * min(self.last_epoch, self.warmup_epochs)
                    / self.warmup_epochs
                )
                for base_lr in self.base_lrs
            ]
        elif self.warmup_epochs < self.last_epoch <= self.steps_before_off_ramp:
            step = self.last_epoch - self.warmup_epochs 
            return [
                base_lr / (1 + (step + self.offset)**self.exponent - self.offset**self.exponent)
                for base_lr in self.base_lrs
            ]
        else:
            lrs_at_end_of_decay = [
                base_lr / (1 + (self.offset + self.max_epochs - self.cooldown_epochs)**self.exponent - self.offset**self.exponent)
                for base_lr in self.base_lrs
            ]
            ramp_epoch = self.last_epoch - self.steps_before_off_ramp - 1
            return [
                decay_lr 
                * (
                    1.0
                    + (self.cooldown_lr_factor - 1.0)
                    * min(self.cooldown_epochs, ramp_epoch)
                    / self.cooldown_epochs
                )
                for decay_lr in lrs_at_end_of_decay
            ]
        
class InverseSqrtLinearWarmupSqrtCooldown(_LRScheduler):
    """Sets the learning rate to follow a linear warmup, then an infinite decay phase,
    followed by a linear off-ramp decay to eta_min."""

    def __init__(
        self,
        optimizer: Optimizer,
        max_epochs: int,
        warmup_epochs: int = 1,
        cooldown_epochs: int = 1,
        warmup_lr_factor: float = 0.1,
        cooldown_lr_factor: float = 0.001,
        step_mult_factor: float = 1,
        last_epoch: int = -1,
        offset: float = 128,
        exponent: float = .5,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            max_epochs (int): Maximum number of iterations
            warmup_epochs (int): Number of epochs for linear warm-up.
            cooldown_epochs (int): Number of epochs for linear cooldown.
            warmup_lr_factor (float): Starting learning rate for warm-up phase.
            cooldown_lr_factor (float): Final learning rate for cooldown phase.
            step_mult_factor (float): Multiplier for all epoch numbers. Default: 1.
            last_epoch (int): Index of the last epoch. Default: -1.
        """
        self.warmup_epochs = int(step_mult_factor * warmup_epochs)
        self.cooldown_epochs = int(step_mult_factor * cooldown_epochs)
        self.max_epochs = int(step_mult_factor * max_epochs)
        self.in_infinite_phase = False
        self.steps_before_off_ramp = self.max_epochs - self.cooldown_epochs
        self.warmup_lr_factor = warmup_lr_factor
        self.cooldown_lr_factor = cooldown_lr_factor
        self.offset = offset
        self.exponent = exponent
        super(InverseSqrtLinearWarmupSqrtCooldown, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate for each phase."""
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self) -> List[float]:
        """Closed-form solution for the scheduler if epoch is passed to the step function."""
        if self.last_epoch <= self.warmup_epochs:
            return [
                base_lr 
                * (
                    self.warmup_lr_factor
                    + (1 - self.warmup_lr_factor)
                    * min(self.last_epoch, self.warmup_epochs)
                    / self.warmup_epochs
                )
                for base_lr in self.base_lrs
            ]
        elif self.warmup_epochs < self.last_epoch <= self.steps_before_off_ramp:
            step = self.last_epoch - self.warmup_epochs 
            return [
                base_lr / (1 + (step + self.offset)**self.exponent - self.offset**self.exponent)
                for base_lr in self.base_lrs
            ]
        else:
            lrs_at_end_of_decay = [
                base_lr / (1 + (self.offset + self.max_epochs - self.cooldown_epochs)**self.exponent - self.offset**self.exponent)
                for base_lr in self.base_lrs
            ]
            
            ramp_epoch = self.last_epoch - self.steps_before_off_ramp - 1
            factor = 1 - (ramp_epoch/self.cooldown_epochs)**.5
            return [factor * base_lr for base_lr in lrs_at_end_of_decay]