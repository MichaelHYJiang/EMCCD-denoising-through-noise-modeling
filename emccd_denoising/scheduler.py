"""Warmup scheduler used by the paper-era trainer."""

from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        if multiplier < 1.0:
            raise ValueError("multiplier must be >= 1")
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [lr * self.multiplier for lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [lr * self.multiplier for lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [lr * (float(self.last_epoch) / self.total_epoch) for lr in self.base_lrs]
        return [
            lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
            for lr in self.base_lrs
        ]

    def step(self, epoch=None, metrics=None):
        if not isinstance(self.after_scheduler, ReduceLROnPlateau):
            if self.finished and self.after_scheduler:
                self.after_scheduler.step(None if epoch is None else epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super().step(epoch)
        else:
            raise NotImplementedError("ReduceLROnPlateau is not used by the paper configuration")
