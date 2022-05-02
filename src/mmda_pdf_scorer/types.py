from typing import Sequence, Tuple, Type, Optional

from pydantic import BaseModel, Field
from mmda.types.span import Span

from .utils import ScoredSentenceRows


class ScholarPhiBoundingBox(BaseModel):
    page: int
    left: float
    width: float
    top: float
    height: float

    @classmethod
    def from_span(cls: Type['ScholarPhiBoundingBox'],
                  span: Span) -> 'ScholarPhiBoundingBox':
        return cls(page=span.box.page,
                   left=span.box.l,
                   width=span.box.w,
                   top=span.box.t,
                   height=span.box.h)


class ScholarPhiAttributes(BaseModel):
    bounding_boxes: Sequence[ScholarPhiBoundingBox]


class ScholarPhiEntityReference(BaseModel):
    id: str
    type: str


class ScoredScholarPhiEntityReference(ScholarPhiEntityReference):
    score: float = 0.0


class ScholarPhiEntity(ScholarPhiEntityReference):
    attributes: ScholarPhiAttributes
    relationships: dict


class ScholarPhiSentenceAttributes(ScholarPhiAttributes):
    text: str


class ScholarPhiSentence(ScholarPhiEntity):
    type: str = 'sentence'
    attributes: ScholarPhiSentenceAttributes
    relationships: dict = Field(default_factory=dict)


class ScholarPhiTermAttributes(ScholarPhiAttributes):
    name: str
    term_type: Optional[str] = None
    definitions: Sequence[str] = Field(default_factory=list)
    definition_texs: Sequence[str] = Field(default_factory=list)
    sources: Sequence[str] = Field(default_factory=list)
    snippets: Sequence[str] = Field(default_factory=list)


class ScholarPhiTermRelationship(BaseModel):
    sentence: ScholarPhiEntityReference
    definition_sentences: Sequence[ScholarPhiEntityReference] = \
        Field(default_factory=list)
    snippet_sentences: Sequence[ScholarPhiEntityReference] = \
        Field(default_factory=list)


class ScholarPhiTerm(ScholarPhiEntity):
    type: str = 'term'
    attributes = ScholarPhiTermAttributes
    relationship = ScholarPhiTermRelationship


def get_sentence_id(span: ScoredSentenceRows):
    return f'S-{span.id or span.uuid}'


def get_term_id(span: ScoredSentenceRows):
    return f'T-{span.id or span.uuid}'


def sentence_to_scholarphi_format(
    span: ScoredSentenceRows
) -> ScholarPhiSentence:

    bounding_boxes = [ScholarPhiBoundingBox.from_span(span)
                      for span in span.spans]

    sentence = ScholarPhiSentence(
        id=get_sentence_id(span),
        attributes=ScholarPhiSentenceAttributes(
            text=span.sentence,
            bounding_boxes=bounding_boxes,
        )
    )

    return sentence


def term_to_scholarphi_format(
    span: ScoredSentenceRows,
    sentences: Sequence[ScholarPhiSentence],
    scores: Sequence[int]
) -> Tuple[ScholarPhiSentence, ScholarPhiTerm]:

    sentence = sentence_to_scholarphi_format(span)

    term = ScholarPhiTerm(
        id=get_term_id(span),
        attributes=ScholarPhiTermAttributes(
            name=span.sentence,
            term_type=None,
            bounding_boxes=sentence.attributes.bounding_boxes,
            definitions=[span.sentence],
            definition_texs=[span.sentence],
            sources=['inline'],
            snippets=[s.attributes.text for s in sentences]
        ),
        relationships=ScholarPhiTermRelationship(
            sentence=ScholarPhiEntityReference(
                id=sentence.id, type=sentence.type
            ),
            snippet_sentences=[
                ScoredScholarPhiEntityReference(
                    id=s.id, type=s.type, score=r
                ) for s, r in zip(sentences, scores)
            ]
        )
    )

    return term, sentence
