from dataclasses import dataclass
from typing import (
    Dict,
    Sequence,
    Optional,
    Union,
    List,
    Any,
    Tuple
)
import logging

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

import springs as sp
from torchmetrics import Metric
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .label_transfer import IdentityLabelTransfer, LabelTransferBase
from ..config import TargetConfig, HuggingFaceModuleConfig

LOGGER = logging.getLogger(__name__)

# this is the type received as input from config
ParameterGroupsType = Sequence[Tuple[Union[str, List[str]], Dict[str, Any]]]

# this is the type for parameters a pyTorch optimizer expects
ParamsType = Sequence[Dict[str, Any]]


@dataclass
class ValidationStepOutput:
    preds: torch.Tensor
    labels: torch.Tensor
    loss: Optional[float] = None


class BaseModule(pl.LightningModule):
    trainer: pl.Trainer
    transfer: LabelTransferBase
    transformer: PreTrainedModel

    def __init__(self,
                 tokenizer: HuggingFaceModuleConfig,
                 transformer: HuggingFaceModuleConfig,
                 transfer: Optional[sp.core.DictConfig] = None,
                 metrics: Optional[Sequence[TargetConfig]] = None,
                 optimizer: Optional[sp.core.DictConfig] = None,
                 scheduler: Optional[sp.core.DictConfig] = None,
                 val_loss_label: str = 'val_loss'):
        super().__init__()

        # this makes sure that input arguments are saved as hyperparameter for
        # this model. Very useful for load_checkpoint. Configurations are all
        # pickable now (espresso-config 0.7 or greater required).
        self.save_hyperparameters()

        self.val_loss_label = val_loss_label

        self.transformer = sp.init(transformer, PreTrainedModel)

        self.transfer = (sp.init.now(transfer, LabelTransferBase) if transfer
                         else IdentityLabelTransfer())

        # need to place metrics in a module dict
        self.metrics = torch.nn.ModuleDict(
            {m: sp.init(c, Metric)
             for m, c in (metrics or {}).items()}   # type: ignore
        )

        # we can't initialize these here or they will create all sort of
        # issue when using distributed training. We instead defer init later.
        self.tokenizer_config = tokenizer
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler

        # self._make_tokenizer: Callable[..., PreTrainedTokenizerBase] = \
        #     sp.init.later(tokenizer, PreTrainedTokenizerBase)
        # self._make_scheduler_fn: Callable[..., _LRScheduler] = \
        #     sp.init.later(scheduler, torch.optim.lr_scheduler._LRScheduler)
        # self._make_

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return sp.init.now(self.tokenizer_config, PreTrainedTokenizerBase)

    def save_pretrained(self, *args, **kwargs):
        self.transformer.save_pretrained(*args, **kwargs)
        self.tokenizer.save_pretrained(*args, **kwargs)

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        """Hides v_num in progress bar as described here:
        https://github.com/PyTorchLightning/pytorch-lightning/issues/1595"""
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict

    def get_num_training_batches(self) -> int:
        if self.trainer.num_training_batches != float('inf'):
            return int(self.trainer.num_training_batches)
        else:
            rank_zero_warn('Requesting dataloader...')

            return len(self.trainer
                       ._data_connector
                       ._train_dataloader_source
                       .dataloader())

    def get_finite_num_of_training_steps(self) -> int:
        num_training_steps = self.trainer.estimated_stepping_batches
        if num_training_steps <= 0 or num_training_steps == float('inf'):
            msg = ('Lightning Module needs to know the total number '
                   'of training steps, but it cannot be determined!')
            raise ValueError(msg)
        return round(num_training_steps)

    def create_parameter_groups(
        self,
        parameter_groups: Optional[ParameterGroupsType] = None
    ) -> ParamsType:
        """Returns groups of parameters that have different optimizer flags.
        Syntax for parameter_groups follows the one used by AllenAI:
        https://docs.allennlp.org/v2.2.0/api/training/optimizers/ """

        parameter_groups = parameter_groups or []

        # this dictionary contains list of all parameters;
        # we subdivide them in groups depending on the parameter groups
        # that have been provided.
        all_params = dict(self.named_parameters())

        # the grouped parameters end up here
        grouped_params = []

        # we iterate over groups
        for names, kw in parameter_groups:
            if isinstance(names, str):
                # some groups are just a single name, so we
                # turn them into a list of names instead.
                names = [names]

            # add a parameter to the list of matching parameters for
            # this group if any of the names in the group is in this
            # parameter. As we add, we pop them from the list of all
            # parameters (what never gets popped is added later)
            matching_params = [all_params.pop(param_name) for param_name
                               in list(all_params.keys())
                               if any(n in param_name for n in names)]
            grouped_params.append({'params': matching_params, **kw})

        # all the parameters that did not end up in a group go into
        # a sort of default group.
        grouped_params.append({'params': list(all_params.values())})

        return grouped_params

    def configure_optimizers(self) -> dict:
        params = self.create_parameter_groups(
            getattr(self.optimizer_config, 'parameter_groups', [])
        )
        optimizer: Optimizer = \
            sp.init.now(self.optimizer_config, Optimizer, params=params)
        configuration: Dict[str, Any] = {'optimizer': optimizer}

        if self.scheduler_config is not None:
            # we first try to determine if the number of steps has been
            # assigned; if not, we use `get_num_training_steps` to estimate.
            kwargs: Dict[str, int] = {}

            num_training_steps: Optional[int] = \
                getattr(self.scheduler_config, 'num_training_steps', None)
            if num_training_steps and num_training_steps < 0:
                kwargs['num_training_steps'] = \
                    self.get_finite_num_of_training_steps()

            if num_training_steps is None:
                kwargs['num_training_steps'] = \
                    round(self.trainer.estimated_stepping_batches)

            num_warmup_steps: Optional[Union[int, float]] = \
                getattr(self.scheduler_config, 'num_warmup_steps', None)
            if num_warmup_steps and isinstance(num_warmup_steps, float):
                if 0 <= num_warmup_steps <= 1:
                    num_warmup_steps *= self.get_finite_num_of_training_steps()
                kwargs['num_warmup_steps'] = round(num_warmup_steps)

            scheduler: _LRScheduler = sp.init.now(
                self.scheduler_config,
                _LRScheduler,
                optimizer=optimizer,
                **kwargs
            )

            configuration['lr_scheduler'] = {
                'scheduler': scheduler,
                'name': 'learning_rate',
                'interval': 'step',
                'frequency': 1
            }

        if self.trainer.is_global_zero:
            LOGGER.info(configuration)

        return configuration
