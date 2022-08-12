from typing import Optional, Sequence, Dict, Optional, Tuple
import logging

import transformers
from transformers.modeling_outputs import TokenClassifierOutput
import torch

from ..config import TargetConfig, TrainConfig

from .base import BaseModule, ValidationStepOutput
import springs as sp

LOGGER = logging.getLogger(__name__)


class TokenClassificationModule(BaseModule):
    def __init__(
        self,
        *args,
        loss: Optional[sp.core.DictConfig] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.loss: torch.nn.Module = (                      # type: ignore
            sp.init(loss) if loss is not None
            else torch.nn.modules.loss.CrossEntropyLoss()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> TokenClassifierOutput:
        out = self.transformer(input_ids=input_ids,
                               attention_mask=attention_mask,
                               **kwargs)

        if labels is not None:
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1

                active_logits = out.logits.view(
                    -1, self.transformer.num_labels
                )

                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(self.loss.ignore_index).type_as(labels)
                )
                out.loss = self.loss(active_logits, active_labels)
            else:
                out.loss = self.loss(
                    out.logits.view(-1, self.transformer.num_labels),
                    labels.view(-1)
                )

        return out

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        # batch is a single stride
        output = self(**batch)
        return output.loss

    def validation_step(self,
                        batch: Dict[str, torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        # batch might be multiple strides

        labels = batch.pop('labels')
        stride_pattern = batch.pop('stride_pattern')

        batch_size, strides, seq_length = labels.shape
        unrolled_batch = {field: tensor.view(-1, tensor.size(-1))
                          for field, tensor in batch.items()}
        output = self(**unrolled_batch, labels=labels)
        rolled_logits = output.logits.view(batch_size, strides, seq_length, -1)

        extracted_preds = []
        extracted_labels = []
        extracted_loss = []

        for i in range(1, stride_pattern.size(0)):
            # TODO: one sample at the time for now; can we do the full batch?

            # Locations where the logits
            valid_logits_loc = torch.nonzero(labels[i] != -100).T
            valid_logits = rolled_logits[i][tuple(valid_logits_loc)]
            valid_labels = labels[i][tuple(valid_logits_loc)]

            valid_strided_locs = torch.nonzero(stride_pattern[i] != - 100).T

            vs_i, vs_j = map(torch.max, valid_strided_locs + 1)
            valid_strided = stride_pattern[i, :vs_i, :vs_j]

            aggregated_logits = (valid_strided.T.type(valid_logits.dtype) @
                                 valid_logits)
            aggregated_labels = (valid_labels.view(-1, 1) *
                                 valid_strided).max(0).values

            extracted_preds.append(aggregated_logits.detach())
            extracted_labels.append(aggregated_labels.detach())
            extracted_loss.append(self.loss(aggregated_logits,
                                            aggregated_labels).detach())

        extracted_preds = torch.cat(extracted_preds).detach()
        extracted_labels = torch.cat(extracted_labels).detach()
        extracted_loss = torch.stack(extracted_loss).detach()

        for metric in self.metrics.values():
            metric(extracted_preds, extracted_labels)

        return extracted_loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self.validation_step(batch=batch,
                                    batch_idx=batch_idx)

    def validation_epoch_end(self, outputs: Sequence[torch.Tensor]):
        val_loss = torch.cat(tuple(outputs)).mean()
        self.log(self.val_loss_label, val_loss,
                 prog_bar=True,
                 sync_dist=True,
                 rank_zero_only=True)
        for metric_name, metric in self.metrics.items():
            self.log(metric_name, metric, rank_zero_only=True)

    def test_epoch_end(self,  outputs: Sequence[torch.Tensor]):
        return self.validation_epoch_end(outputs=outputs)


class SimpleTokenClassificationModule(BaseModule):
    loss: torch.nn.modules.loss._Loss

    def __init__(
        self,
        *args,
        loss: Optional[TargetConfig] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.loss = (sp.init(loss, torch.nn.modules.loss._Loss) or
                     torch.nn.modules.loss.CrossEntropyLoss())

    def calculate_loss(
        self: 'SimpleTokenClassificationModule',
        out: TokenClassifierOutput,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_labels: Optional[int] = None,
    ) -> TokenClassifierOutput:

        num_labels = num_labels or \
                     int(self.transformer.num_labels)   # type: ignore

        if labels is not None:
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = out.logits.view(-1, num_labels)

                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(self.loss.ignore_index).type_as(labels)
                )
                out.loss = self.loss(active_logits, active_labels)

            else:
                out.loss = self.loss(out.logits.view(-1, num_labels),
                                     labels.view(-1))
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> TokenClassifierOutput:

        out = self.transformer(input_ids=input_ids,
                               attention_mask=attention_mask,
                               **kwargs)
        return self.calculate_loss(out=out,
                                   attention_mask=attention_mask,
                                   labels=labels)

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:

        # batch is a single stride
        output = self(**batch)
        self.log('train/loss',
                 output.loss,
                 on_step=True,
                 prog_bar=False,
                 sync_dist=True)
        return output.loss

    def validation_step(self,
                        batch: Dict[str, torch.Tensor],
                        batch_idx: int) -> ValidationStepOutput:

        output = self(**batch)

        locs = batch['labels'] != self.loss.ignore_index
        extracted_labels = batch['labels'][locs]
        extracted_preds = output.logits[locs]

        # this takes care of mapping
        extracted_preds = self.transfer(extracted_preds)

        if extracted_labels.size(0):
            for metric in self.metrics.values():
                metric(extracted_preds, extracted_labels)

        return ValidationStepOutput(loss=output.loss,
                                    labels=extracted_labels,
                                    preds=extracted_preds)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self.validation_step(batch=batch,
                                    batch_idx=batch_idx)

    def validation_epoch_end(
        self,
        outputs: Sequence[ValidationStepOutput],
        prefix: str = 'validation'
    ):
        val_loss = torch.stack([o.loss for o in outputs]).mean()
        self.log(self.val_loss_label,
                 val_loss,
                 prog_bar=True,
                 sync_dist=True,
                 rank_zero_only=True)
        for metric_name, metric in self.metrics.items():
            if hasattr(metric, 'average') and metric.average is None:
                for i, m in enumerate(metric.compute()):
                    self.log(f'{prefix}/{metric_name}/{i}',
                             m,
                             rank_zero_only=True)
            else:
                self.log(f'{prefix}/{metric_name}',
                         metric,
                         rank_zero_only=True)

    def test_epoch_end(self,  outputs: Sequence[torch.Tensor]):
        return self.validation_epoch_end(outputs=outputs, prefix='text')


class SaliencyClassificationModule(SimpleTokenClassificationModule):
    def __init__(
        self,
        *args,
        threshold: float = 0.5,
        transfer: Optional[sp.core.DictConfig] = None,
        val_loss_label: str = 'val_loss',
        metrics: Optional[Sequence[TargetConfig]] = None,
        num_labels: int = 2,
        **kwargs
    ):

        super().__init__(transformer=None,
                         tokenizer=None,
                         transfer=transfer,
                         metrics=metrics,
                         val_loss_label=val_loss_label)

        self.num_labels = num_labels
        self.threshold = threshold

    def forward(
        self,
        # input_ids: torch.Tensor,
        # attention_mask: torch.Tensor,
        salience: torch.Tensor,
        *args,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> TokenClassifierOutput:

        # rescale salience to [-1 to 1] using threshold. Assumes that
        # salience is already normalized to [0 to 1].
        #
        # For value above the threshold, we first subtract the threshold,
        # then divide by 1 - threshold. For example, if threshold is 0.3,
        # then an original salience value of 0.5 will be rescaled to
        # ( 0.5 - 0.3 ) / 0.7 = 0.285...; note how, using this formula,
        # 1 gets mapped to 1. Using the formula for values below threshold,
        # 0.2 would be mapped to -0.333...; note how, using this formula,
        # 0 gets mapped to -1.
        salience = torch.where(
            salience > self.threshold,
            (salience - self.threshold) / (1 - self.threshold),
            (salience - self.threshold) / self.threshold,
        )

        logits = torch.stack((-salience, salience), axis=-1)
        out = TokenClassifierOutput(logits=logits.squeeze(-1))

        return self.calculate_loss(out=out,
                                   labels=labels,
                                   num_labels=self.num_labels)
