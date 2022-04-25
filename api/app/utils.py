from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
from typing import Literal, Sequence

from mmda.types.annotation import SpanGroup
from mmda.types.span import Span
from mmda.types.box import Box


@dataclass
class SentenceRows(SpanGroup):
    rows: Sequence[SpanGroup] = field(default_factory=list)

    @property
    def sentence(self):
        return ' '.join(''.join(symbol.rstrip('-') for symbol in word.symbols)
                        for word in self.words)


def partition_row_on_token(row: SpanGroup,
                           token: SpanGroup,
                           direction: Literal['before', 'after']) -> SpanGroup:
    """Partition a row on a given token; keep either everything before the
    token, or everything after it (token is always included)"""

    assert len(row.rows) == 1, "`row` is not a single row"
    assert len(token.tokens) == 1, "`token` is not a single token"
    assert token in row.tokens, "token is not part of this row"

    if direction == 'after':
        row_tokens = [t for t in row.tokens if t.start >= token.start]
    elif direction == 'before':
        row_tokens = [t for t in row.tokens if t.end <= token.end]
    else:
        raise ValueError(f'Direction "{direction}" is not "before" or "after"')

    new_row_boxes = [span.box for t in row_tokens for span in t]
    new_span = Span(start=row_tokens[0][0].start,
                    end=row_tokens[-1][0].end,
                    box=Box.small_boxes_to_big_box(new_row_boxes))
    new_row = SpanGroup(spans=[new_span])
    new_row.attach_doc(row.doc)

    return new_row


def slice_block_from_tokens(block: SpanGroup,
                            bos_token: SpanGroup,
                            eos_token: SpanGroup) -> SentenceRows:
    """Given a block with rows, return a slice that contains only rows
    between start_token and end_token included."""

    rows_accumulator = []
    for row in block.rows:
        if row.end < bos_token.start:
            # this row is completely before start token
            continue
        elif row.start > eos_token.end:
            # this row is completely after end token
            continue
        elif bos_token.start >= row.start and bos_token.end <= row.end:
            # the start token is part of this row; we need to
            # partition this row so that only part of it is included
            # in the accumulator
            row = partition_row_on_token(
                row=row, token=bos_token, direction='after'
            )
        elif eos_token.start >= row.start and eos_token.end <= row.end:
            # same as before, but this time for the end token
            row = partition_row_on_token(
                row=row, token=eos_token, direction='before'
            )

        # in all other cases, the row is completely within
        rows_accumulator.append(row)

    new_spans = list(chain.from_iterable(r.spans for r in rows_accumulator))
    type_ = getattr(block.box_group, 'type', None)

    new_rows = SentenceRows(spans=new_spans, rows=rows_accumulator, type=type_)
    new_rows.attach_doc(rows_accumulator[0].doc)

    return new_rows
