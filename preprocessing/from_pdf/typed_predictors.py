import re
from typing import Iterable, List, Optional, Sequence, Set, Tuple, Dict

from mmda.predictors.base_predictors.base_predictor import BasePredictor
from mmda.types.annotation import BoxGroup, SpanGroup, Annotation, Span
from mmda.types.document import Document
from mmda.types.names import Words, Sentences, Blocks

from .layout_tools import (
    intersect_span_groups,
    span_is_fully_contained,
    box_group_from_span_group,
)
from .types import TypedBlocks


def make_typed_span_group(
    spans: List[Span],
    document: Document,
    type_: Optional[str] = None,
    id_: Optional[int] = None,
    add_text: bool = True,
) -> SpanGroup:
    typed_sg = SpanGroup(
        spans=spans,
        id=id_,
        type=type_ or 'Other',
    )
    typed_sg.box_group = box_group_from_span_group(
        span_group=typed_sg, doc=document
    )
    if add_text:
        typed_sg.text = ' '.join(
            str(word.text) for word in
            document.find_overlapping(typed_sg, Words)
        )
    return typed_sg


class TypedBlockPredictor(BasePredictor):
    REQUIRED_BACKENDS = None                        # type: ignore
    REQUIRED_DOCUMENT_FIELDS = [Blocks, Words]      # type: ignore

    Text = 'Text'
    Title = 'Title'
    ListType = 'List'
    Table = 'Table'
    Figure = 'Figure'
    Other = 'Other'
    RefApp = 'ReferencesAppendix'
    Abstract = 'Abstract'
    Preamble = 'Preamble'

    def _get_block_words(self,
                         doc: Document,
                         block: SpanGroup) -> Sequence[str]:
        words = doc.find_overlapping(block, Words)
        return [str(w.text if w.text else '') for w in words]

    def _tag_abstract_blocks(self, doc: Document, blocks: List[SpanGroup]):
        in_abstract = False
        abstract_position = -1
        for i, block in enumerate(blocks):
            if (
                block.box_group is not None
                and block.box_group.type == self.Title
            ):
                sec_type = re.sub(
                    r'^(\d|[\.])+\s+', '',
                    # remove leading section numbers if present
                    ' '.join(self._get_block_words(doc, block))
                ).lower()
                if abstract_position >= 0:
                    break

                # HEURISTIC only check for match in the first 20 chars or so
                if 'abstract' in sec_type[:20]:
                    abstract_position = i
                    in_abstract = True

            if in_abstract:
                block.type = self.Abstract

        # mark everything before first abstract
        if abstract_position > 0:
            for block in blocks[:abstract_position]:
                block.type = self.Preamble
        elif (
            abstract_position == 0 and
            (abstract_start := blocks[0].start) > 0
        ):
            # make a preamble block if the first recognized block has
            # zero index but is not at the first position in the document
            preamble_sg = make_typed_span_group(
                spans=[Span(start=0, end=abstract_start)],
                document=doc,
                type_=self.Preamble,
                add_text=False,
            )
            blocks.insert(0, preamble_sg)

    def _tag_references_blocks(
        self,
        doc: Document,
        blocks: Sequence[SpanGroup]
    ) -> None:
        in_references = False
        for block in blocks:
            if (
                block.box_group is not None and (
                    block.box_group.type == self.Title
                    # HEURISTIC sometimes the title of the references section
                    # is incorrectly tagged as a list, so we check for that
                    # type as well.
                    or block.box_group.type == self.ListType
                )
            ):
                sec_type = re.sub(
                    # remove leading section numbers if present
                    r'^(\d|[\.])+\s+', '',
                    ' '.join(self._get_block_words(doc, block))
                ).lower()
                # HEURISTIC only check for match in the first 20 chars or so
                if 'references' in sec_type[:20]:
                    in_references = True

            if in_references:
                block.type = self.RefApp

    def _create_typed_blocks(self, doc: Document) -> Sequence[SpanGroup]:
        cur_blocks: List[SpanGroup] = getattr(doc, 'blocks', [])
        new_blocks: List[SpanGroup] = []

        for block in cur_blocks:
            block_type = None

            if len(block.spans) < 1 or block.box_group is None:
                continue
            elif (
                block.box_group.type == self.Text or
                block.box_group.type == self.ListType
            ):
                block_type = block.box_group.type
            elif block.box_group.type == self.Title:
                cur_sents: Iterable[SpanGroup] = block.sents   # type: ignore

                sents = [
                    sent for sent in cur_sents if
                    span_is_fully_contained(block, sent)
                ]

                if len(sents) >= 2:
                    # HEURISTIC: something tagged as a title with at
                    # least two fully contained sentences is probably a text
                    block_type = self.Text
                else:
                    block_type = self.Title
            else:
                block_type = block.box_group.type or self.Other

            new_blocks.append(
                SpanGroup(
                    spans=block.spans,
                    id=len(new_blocks),
                    type=str(block_type),
                    box_group=BoxGroup(
                        boxes=block.box_group.boxes,
                        type=block.box_group.type
                    )
                )
            )
        return new_blocks

    def predict(self, document: Document) -> Sequence[Annotation]:
        typed_blocks = self._create_typed_blocks(document)
        self._tag_abstract_blocks(document, typed_blocks)
        self._tag_references_blocks(document, typed_blocks)
        return typed_blocks


class TypedSentencesPredictor(BasePredictor):
    REQUIRED_BACKENDS = None                                # type: ignore
    REQUIRED_DOCUMENT_FIELDS = [TypedBlocks, Sentences]     # type: ignore

    CONTENT_TYPES: Set[str] = {
        TypedBlockPredictor.Text,
        TypedBlockPredictor.ListType,
        TypedBlockPredictor.Abstract,
    }
    LAYOUT_TYPES: Set[str] = {
        TypedBlockPredictor.Title,
        TypedBlockPredictor.Table,
        TypedBlockPredictor.Figure,
        TypedBlockPredictor.Preamble
    }

    def predict(self, document: Document) -> List[SpanGroup]:
        # The logic here is as follows:
        # 1. We iterate over each typed block.
        # 2. For each typed block, we look at all sentences overlapping with
        #    the block.
        #      2a. Current typed block is a title, table, or figure; in that
        #          case, we create the minimum sentence overlapping with the
        #          block and add that to the typed sentences.
        #      2b. If a sentence ends, but not starts in the current block, we
        #          ignore it; a previous block already added it in the typed
        #          sentences.
        #             - An exception to this is if the sentence belongs to no
        #               other blocks, and the current block is a Text, List, or
        #               Abstract.
        #      2c. If a sentence starts and ends in the current block we add it
        #          to the typed sentences unless it has been added already.
        #      2d. If a sentence starts and does not end in the current block,
        #          we add it to the typed sentences assuming it is part of the
        #          current block.
        # 3. If a sentence belongs to no blocks, we add it to the typed
        #    sentences with a type of Other.

        typed_block: SpanGroup
        sent: SpanGroup
        typed_sents: List[SpanGroup] = []

        # typed_sents_loc: Dict[int, int] = {}
        # sentences_that_have_already_been_sliced: Set[Tuple[int, int]] = set()

        # This dictionary contains the index of the last position a sentence
        # was sliced on; the key is a tuple of (start_of_unsliced,
        # end_of_unsliced). If a sentence is not sliced, then the value
        # is the end position.
        typed_sents_ends: Dict[Tuple[int, int], int] = {}

        print('\n\n\n')

        for typed_block in document.typed_blocks:   # type: ignore
            is_content_block = typed_block.type in self.CONTENT_TYPES

            for sent in typed_block.sents:          # type: ignore
                if typed_block.type in self.LAYOUT_TYPES:
                    # CASE 1: This sentence is part of a title, table,
                    #         figure, or preamble. Therefore, because it is
                    #         part of a visual block, we want to make
                    #         this sentence as tight as possible, meaning not
                    #         letting it span across other blocks.
                    tight_spans = intersect_span_groups(typed_block, sent)
                    tight_sg = make_typed_span_group(
                        spans=tight_spans,
                        document=document,
                        type_=typed_block.type,
                        id_=len(typed_sents)
                    )
                    add_to_typed_sentences = (
                        (
                            key := (sent.start, sent.end)
                        ) not in typed_sents_ends
                        or typed_sents_ends[key] < tight_sg.end
                    )

                    if add_to_typed_sentences:
                        typed_sents_ends[key] = tight_sg.end
                        typed_sents.append(tight_sg)
                        # sentences_that_have_already_been_sliced.add(
                        #     (sent.start, sent.end)
                        # )
                        # print('1', tight_sg.type, tight_sg.start, tight_sg.end, tight_sg.text)

                    # if tight_sg.start == 19009:
                    #     import ipdb
                    #     ipdb.set_trace()
                    # # # # # # END OF CASE 1 # # # # # #

                elif sent.start < typed_block.start:
                    # CASE 2: This sentence starts before the current block.
                    #         We have a few more checks to do before being able
                    #         to add it to the typed sentences.

                    previous_blocks_are_all_visual = all(
                        b.type in self.LAYOUT_TYPES
                        for b in sent.typed_blocks
                        if b.start < typed_block.start  # type: ignore
                    )
                    already_sliced = (
                        (key := (sent.start, sent.end)) in typed_sents_ends
                    )

                    if is_content_block and (
                        previous_blocks_are_all_visual or already_sliced
                    ):
                        # CASE 2a: We add this sentence to the one extracted
                        #          from this block because (a) the current
                        #          block is a content block, meaning text,
                        #          list, or abstract, and (b) the sentence
                        #          belongs to all visual blocks except the
                        #          current block.
                        tight_spans = intersect_span_groups(typed_block, sent)
                        tight_sg = make_typed_span_group(
                            spans=tight_spans,
                            document=document,
                            type_=typed_block.type,
                            id_=len(typed_sents)
                        )
                        if typed_sents_ends.get(key, -1) < tight_sg.end:
                            typed_sents_ends[key] = tight_sg.end
                            typed_sents.append(tight_sg)
                            # print('2a', tight_sg.type, tight_sg.start, tight_sg.end, tight_sg.text)

                        # if tight_sg.start not in typed_sents_loc:
                        #     typed_sents_loc[tight_sg.start] = tight_sg.end
                        #     typed_sents.append(tight_sg)
                        #     sentences_that_have_already_been_sliced.add(
                        #         (sent.start, sent.end)
                        #     )
                        #     print('2', tight_sg.type, tight_sg.start, tight_sg.end, tight_sg.text)
                        # # # # # # END OF CASE 2a # # # # # #

                        # if (
                        #     sent_key := (tight_sg.start, tight_sg.end)
                        # ) not in typed_sents:
                        #     typed_sents[sent_key] = tight_sg

                elif (key := (sent.start, sent.end)) not in typed_sents_ends:
                    # CASE 3: This sentence starts in the current block.
                    #         We add it to the typed sentences unless it has
                    #         been added already.
                    new_sg = make_typed_span_group(
                        spans=sent.spans,
                        type_=typed_block.type,
                        document=document,
                        id_=len(typed_sents)
                    )
                    typed_sents_ends[key] = new_sg.end
                    typed_sents.append(new_sg)
                    # print('3', new_sg.type, new_sg.start, new_sg.end, new_sg.text)

        # import ipdb
        # ipdb.set_trace()

        # for sent in document.sents:     # type: ignore
        #     if (sent_key := (sent.start, sent.end)) not in typed_sents:
        #         # case 3
        #         typed_sents[sent_key] = make_typed_span_group(
        #             spans=sent.spans,
        #             document=document,
        #             id_=len(typed_sents)
        #         )
        # typed_sentences_list = sorted(typed_sents.values(), key=lambda x: x.start)

        # print('\n\n\n')

        # import ipdb; ipdb.set_trace()

        return typed_sents

        ################################################################

        # sent: SpanGroup
        # for sent in document.sents:     # type: ignore

        #     sent_extracted_spans: List[SpanGroup] = []
        #     spans_seen_so_far: List[Span] = []

        #     block: SpanGroup
        #     for block in sent.typed_blocks:     # type: ignore
        #         spans_inter = intersect_span_groups(sent, block)

        #         new_sent_span_group = SpanGroup(
        #             spans=spans_inter,
        #             id=(typed_sents_cnt := typed_sents_cnt + 1),
        #             type=block.type
        #         )
        #         new_sent_span_group.box_group = box_group_from_span_group(
        #             span_group=new_sent_span_group, doc=document
        #         )
        #         new_sent_span_group.text = ' '.join(
        #             str(word.text) for word in
        #             document.find_overlapping(new_sent_span_group, Words)
        #         )
        #         import ipdb
        #         ipdb.set_trace()

        #         # sent_extracted_spans.append(

        #         # )
        #         spans_seen_so_far.extend(spans_inter)

        #     # last step here is to add all sentences that are *not* part of
        #     # any block. We do so by calling make_span_group_from_spans to
        #     # get all segments that overlap with the spans seen so far, and
        #     # then taking only the ones that are part of the source sentence.
        #     segments = make_span_groups_segments(sent, spans_seen_so_far)
        #     for segment in segments.src:
        #         pass

        ################################################################

        #     sent_blocks_types: Set[Optional[str]] = \
        #         set(b.type for b in sent.typed_blocks)  # type: ignore

        #     if len(sent_blocks_types) <= 1:
        #         sent_type = next(iter(sent_blocks_types), None)
        #         typed_sents.append(
        #             SpanGroup(
        #                 spans=sent.spans,
        #                 id=len(typed_sents),
        #                 type=sent_type,
        #                 text=' '.join(str(w.text) for w in sent.words),
        #                 box_group=box_group_from_span_group(sent),
        #             )
        #         )
        #     else:
        #         cur_blocks: Iterable[SpanGroup] = \
        #             sent.typed_blocks  # type: ignore

        #         for block in cur_blocks:
        #             new_spans = intersect_span_groups(sent, block)

        #             new_sent = SpanGroup(
        #                 spans=new_spans,
        #                 id=len(typed_sents),
        #                 type=block.type,
        #             )
        #             new_sent.text = ' '.join(
        #                 str(w.text if w.text else '') for w in
        #                 document.find_overlapping(new_sent, Words)
        #             )
        #             new_sent.box_group = box_group_from_span_group(
        #                 new_sent, doc=document
        #             )

        #             typed_sents.append(new_sent)

        #     print(' '.join(sent.symbols))
        #     import ipdb
        #     ipdb.set_trace()

        return typed_sents
