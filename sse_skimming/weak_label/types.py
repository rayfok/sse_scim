from dataclasses import dataclass
from typing import Optional
from enum import IntEnum


@dataclass
class Instance:
    id: int
    text: str
    type: Optional[str] = ""
    section: Optional[str] = ""
    block_id: Optional[str] = -1
    span_start: Optional[int] = -1
    span_end: Optional[int] = -1
    doc_id: str = ""
    label: Optional[str] = None
    score: Optional[int] = None


class Label(IntEnum):
    ABSTAIN = -1
    NONSIG = 0
    METHOD = 1
    RESULT = 2
    NOVELTY = 3
