
from dataclasses import dataclass, field, fields
from itertools import chain
from re import I
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


@dataclass
class ScoredSentenceRows(SentenceRows):
    score: float = None
    query: SentenceRows = None

    @classmethod
    def from_sentence(cls,
                      query: SentenceRows,
                      score: float,
                      sentence: SentenceRows) -> 'ScoredSentenceRows':
        sentence_data = {
            f.name: getattr(sentence, f.name) for f in fields(sentence)
            # we want a fresh new uuid && we will attach dataset later
            if f.name != 'uuid' and f.name != 'doc' and f.name != 'id'
        }
        scored_sentence = cls(score=score,
                              id=sentence.id or sentence.uuid,
                              query=query,
                              **sentence_data)
        scored_sentence.attach_doc(sentence.doc)

        return scored_sentence


def partition_row_on_token(row: SpanGroup,
                           token: Span,
                           direction: Literal['before', 'after']) -> SpanGroup:
    """Partition a row on a given token; keep either everything before the
    token, or everything after it (token is always included)"""

    if len(row.rows) != 1:
        raise ValueError("`row` is not a single row")
    if token.start < row.start or token.end > row.end:
        raise ValueError("token is not part of this row")

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

        if row.start > eos_token.end:
            # this row is completely after end token
            continue

        if bos_token[0].start >= row.start and bos_token[0].end <= row.end:
            # the start token is part of this row; we need to
            # partition this row so that only part of it is included
            # in the accumulator
            #
            # In the case of multi tokens spans (which is the case
            # for words that span over multiple rows), we only care
            # about the first one.
            row = partition_row_on_token(
                row=row, token=bos_token[0], direction='after'
            )

        if eos_token[-1].start >= row.start and eos_token[-1].end <= row.end:
            # same as before, but this time for the end token
            #
            # In the case of multi tokens spans, we only care
            # about the last one.
            row = partition_row_on_token(
                row=row, token=eos_token[-1], direction='before'
            )

        # in all other cases, the row is completely within
        rows_accumulator.append(row)

    new_spans = chain.from_iterable(r.spans for r in rows_accumulator)
    type_ = getattr(block.box_group, 'type', None)

    new_rows = SentenceRows(spans=list(new_spans),
                            rows=rows_accumulator,
                            type=type_)
    new_rows.attach_doc(rows_accumulator[0].doc)

    return new_rows
