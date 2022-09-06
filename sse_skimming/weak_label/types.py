from dataclasses import dataclass
from typing import Optional


@dataclass
class Instance:
    id: str
    type: str
    text: str
    section: str
    block_id: str
    span_start: int
    span_end: int
    label: Optional[str] = None
    score: Optional[int] = None
