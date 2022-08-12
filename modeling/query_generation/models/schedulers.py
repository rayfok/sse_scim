from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class LinearScheduleWithWarmup(LambdaLR):
    def _lr_lambda(self, current_step: int):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        return max(
            0.0, float(self.num_training_steps - current_step) /
            float(max(1, self.num_training_steps - self.num_warmup_steps))
        )

    def __init__(self,
                 optimizer: Optimizer,
                 num_warmup_steps: int,
                 num_training_steps: int,
                 last_epoch: int = -1
                 ) -> None:
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        super().__init__(optimizer, self._lr_lambda, last_epoch)
