from typing import Sequence, Dict, Optional
import logging

import numpy as np
import transformers
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

from .base import BaseModule, ValidationStepOutput
from .metrics import MetricWrapper
from ..config import TargetConfig


LOGGER = logging.getLogger(__name__)


class Sequence2SequenceModule(BaseModule):
    def __init__(self,
                 *args,
                 metrics: Optional[Sequence[TargetConfig]] = None,
                 eval_accumulate_multiple_refs: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # Use metric wrapper to manage different aspects of metric calculation
        self.metrics = MetricWrapper.from_configs_dict(
            configs_dict=metrics,
            eval_accumulate_multiple_refs=eval_accumulate_multiple_refs
        )

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs
    ) -> transformers.modeling_outputs.Seq2SeqLMOutput:
        return self.transformer(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                                **kwargs)

    def generate(
        self,
        labels: torch.Tensor = None,
        return_dict_in_generate: bool = True,
        **kwargs
    ) -> transformers.generation_utils.GreedySearchDecoderOnlyOutput:
        # TODO: implement non-greedy decoding
        if not return_dict_in_generate:
            rank_zero_warn('Overwriting `return_dict_in_generate` to True.')
        if labels is not None:
            rank_zero_warn('Labels are ignored during generation.')

        return self.transformer.generate(**kwargs,
                                         return_dict_in_generate=True)

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        output = self(**batch)
        return output.loss

    def validation_step(self,
                        batch: Dict[str, torch.Tensor],
                        batch_idx: int,
                        add_val_loss: bool = True) -> ValidationStepOutput:
        # we run both forward and generation pass to get loss
        # value on the sequence
        if add_val_loss:
            forced_dec_output = self(**batch)
            val_loss = float(forced_dec_output.loss)
        else:
            val_loss = None

        generation_output = self.generate(**batch)

        # this replaces the -100 used for padding during loss calculation
        # with pad_token_id, which then can be removed by the tokenizer
        # when decoding
        labels = torch.where(batch['labels'] != -100,
                             batch['labels'],
                             self.transformer.config.pad_token_id)

        return ValidationStepOutput(
            val_loss=val_loss,
            preds=generation_output.sequences.detach().to('cpu'),
            labels=labels.detach().to('cpu')
        )

    def test_step(self,
                  batch: Dict[str, torch.Tensor],
                  batch_idx: int) -> ValidationStepOutput:
        return self.validation_step(batch=batch,
                                    batch_idx=batch_idx,
                                    add_val_loss=False)

    def validation_epoch_end(self,
                             outputs: Sequence[ValidationStepOutput],
                             add_val_loss: bool = True):
        # we don't want to instantiate the tokenizer more than necessary
        tokenizer = self.tokenizer()

        for metric_name, metric in self.metrics.items():
            metric_output = metric(batches=outputs, tokenizer=tokenizer)

            for k, v in metric_output.items():
                self.log(f'{metric_name}/{k}', v, prog_bar=False)

        if add_val_loss:
            val_loss = np.mean([batch.val_loss for batch in outputs])
            self.log({f'{self.val_loss_label}'}, val_loss, prog_bar=True)

    def test_epoch_end(self,  outputs: Sequence[ValidationStepOutput]):
        return self.validation_epoch_end(outputs=outputs, add_val_loss=False)
