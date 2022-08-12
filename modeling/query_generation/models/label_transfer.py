from typing import Dict, Literal, List
from abc import ABC, abstractmethod

import torch


class LabelTransferBase(ABC):
    @abstractmethod
    def __call__(self, preds: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class IdentityLabelTransfer(LabelTransferBase):
    def __call__(self, preds: torch.Tensor) -> torch.Tensor:
        return preds


class MapLabelTransfer(LabelTransferBase):
    def __init__(
        self,
        mapping: Dict[int, int],
        strategy: Literal['max', 'avg', 'min'],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert all(isinstance(key, int) for key in mapping.keys()), \
            'All keys for mapping must be integers'
        assert all(isinstance(val, int) for val in mapping.values()), \
            'All values for mapping must be integers'
        assert strategy in ['max', 'avg', 'min'], \
            'Strategy must be one of `max`, `avg`, `min`'

        num_to_values = len(set(mapping.values()))
        max_pos = max(mapping.values())
        assert num_to_values == (max_pos + 1), (
            f'Values for mapping must be contiguous; got {num_to_values}'
            f' but the max position is {max_pos}'
        )

        self.mapping: List[List[int]] = [
            [] for _ in range(max(mapping.values()) + 1)
        ]
        for key, val in mapping.items():
            self.mapping[val].append(key)

        self.strategy = strategy

    def _op(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.strategy == 'max':
            return tensor.max(dim=-1).values
        elif self.strategy == 'avg':
            return tensor.mean(dim=-1)
        elif self.strategy == 'min':
            return tensor.min(dim=-1).values
        else:
            raise ValueError(f'Unknown strategy: {self.strategy}')

    def __call__(self, preds: torch.Tensor) -> torch.Tensor:
        to_group_preds = [preds[:, old_idxs] for old_idxs in self.mapping]

        mapped_preds = torch.stack(
            [self._op(preds) for preds in to_group_preds],
            dim=-1
        )
        return mapped_preds
