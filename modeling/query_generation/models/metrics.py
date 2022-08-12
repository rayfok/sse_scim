from typing import (Sequence, Union, Callable, Dict, Any,
                    Optional, Type)
import torch
import torchmetrics
import springs as sp
import transformers

from torchmetrics.metric import Metric

from ..config import TargetConfig

RawMetricObjType = Union[torchmetrics.Metric, Callable, Dict]
MetricConfigType = Union[Dict[str, Any], TargetConfig]


class MetricWrapper:
    MULTIPLE_REFS_ALLOWED = (
        'torchmetrics.functional.text.rouge.rouge_score',
        'torchmetrics.text.rouge.ROUGEScore'
    )

    @classmethod
    def from_configs_dict(
        cls: Type['MetricWrapper'],
        configs_dict: Dict[str, MetricConfigType],
        *args,
        **kwargs
    ) -> Dict[str, 'MetricWrapper']:

        return {
            metric_name: cls(*args,
                             metric_name=metric_name,
                             metric_config=metric_config,
                             **kwargs)
            for metric_name, metric_config in configs_dict.items()
        }

    def __init__(
        self: 'MetricWrapper',
        metric_name: str,
        metric_config: MetricConfigType,
        eval_accumulate_multiple_refs: bool = False,
        batch_preds_field_name: str = 'preds',
        batch_targets_field_name: str = 'labels'
    ) -> None:
        self.name = metric_name
        self.config = metric_config

        try:
            self.fn = sp.init.now(config=metric_config)
            self.partial = False
        except TypeError:
            self.fn = sp.init.later(config=metric_config)
            self.partial = True

        self.multiple_refs = (eval_accumulate_multiple_refs and
                              self._is_multiple_ref_allowed(metric_config))
        self.is_text = self._is_text_metric(metric_config)

        self.preds_name = batch_preds_field_name
        self.targets_name = batch_targets_field_name

    @staticmethod
    def _get_target(m: RawMetricObjType):
        if isinstance(m, ConfigNode) or isinstance(m, dict):
            return m.get('_target_', '')
        else:
            return m.__module__

    @classmethod
    def _is_text_metric(cls, m: RawMetricObjType) -> bool:
        target = cls._get_target(m)

        return (isinstance(m, RawMetricObjType.__args__) and (
            target.startswith('torchmetrics.text') or
            target.startswith('torchmetrics.functional.text')))

    @staticmethod
    def _get_element_in_batch(batch: Any, key: str) -> Any:
        if isinstance(batch, dict):
            return batch.get(key)
        else:
            return getattr(batch, key)

    def get_targets(self, batch: Any) -> Any:
        return self._get_element_in_batch(batch, self.targets_name)

    def get_preds(self, batch: Any) -> Any:
        return self._get_element_in_batch(batch, self.preds_name)

    @classmethod
    def _is_multiple_ref_allowed(cls, m: RawMetricObjType) -> bool:
        return cls._get_target(m) in cls.MULTIPLE_REFS_ALLOWED

    def __call__(self,
                 batches: Sequence[Any],
                 tokenizer: Optional[transformers.PreTrainedTokenizer] = None):
        if self.is_text and tokenizer is None:
            ValueError("You must provide a tokenizer for text metrics")

        if self.is_text:
            # is a metric that works on text, so we have to decode!
            def decode_fn(seq):
                return tokenizer.batch_decode(seq, skip_special_tokens=True)
        else:
            # a numeric metric, so the decode operation os a simple no-op
            def decode_fn(seq):
                return seq

        # slightly different logic depending on whether we have multiple
        # refs or not
        if self.multiple_refs:
            # if we have multiple refs for the same generation, we accumulate
            # them here
            prediction_grouping_dict = {}
            for batch in batches:
                decoded_targets = decode_fn(self.get_targets(batch))
                decoded_preds = decode_fn(self.get_preds(batch))
                for p, t in zip(decoded_preds, decoded_targets):
                    prediction_grouping_dict.setdefault(p, []).append(t)

            preds, targets = map(list, zip(*prediction_grouping_dict.items()))
        else:
            targets, preds = [], []

            for batch in batches:
                decoded_targets = decode_fn(self.get_targets(batch))
                decoded_preds = decode_fn(self.get_preds(batch))
                targets.extend(decoded_targets)
                preds.extend(decoded_preds)

        metric_output = self.fn(preds=preds, target=targets)
        metric_output = metric_output if self.partial else self.fn.compute()
        return metric_output


class CountMetric(Metric):
    count: torch.Tensor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state('count',
                       default=torch.zeros([], dtype=torch.long),
                       dist_reduce_fx=None)

    def update(self,
               preds: torch.Tensor,
               target: torch.Tensor) -> None:
        self.count += target.size(0)

    def compute(self) -> torch.Tensor:
        return self.count
