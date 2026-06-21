import math
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler


class MultiStepRestartLR(_LRScheduler):
    """MultiStep with restarts learning rate scheme."""

    def __init__(self, optimizer, milestones, gamma=0.1, restarts=(0, ), restart_weights=(1, ), last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.restarts = restarts
        self.restart_weights = restart_weights
        assert len(self.restarts) == len(self.restart_weights), 'restarts and their weights do not match.'
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma**self.milestones[self.last_epoch] for group in self.optimizer.param_groups]


def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.

    Args:
        iteration (int): Current iteration.
        cumulative_period (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i


class CosineAnnealingRestartLR(_LRScheduler):
    """Cosine annealing with restarts. Each cycle restarts at `restart_weights[i] * base_lr`."""

    def __init__(self, optimizer, periods, restart_weights=(1, ), eta_min=0, last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        assert len(self.periods) == len(self.restart_weights), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [sum(self.periods[:i + 1]) for i in range(len(self.periods))]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]

        return [
            self.eta_min + current_weight * 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]
