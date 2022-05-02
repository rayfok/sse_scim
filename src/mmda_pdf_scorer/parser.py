
from dataclasses import dataclass
from enum import IntEnum
import logging
from pathlib import Path
from typing import Optional, OrderedDict, Sequence, Tuple, Union
from espresso_config import cli, instantiate, configure_logging
import torch
import warnings
import tqdm

from mmda.types.document import Document
from mmda.types.annotation import SpanGroup
from mmda.rasterizers.rasterizer import Rasterizer
from mmda.parsers.parser import Parser
from mmda.predictors.base_predictors.base_predictor import BasePredictor

from spacy.tokens import Doc

from .config import MmdaParserConfig, CliMmdaParserConfig
from .utils import slice_block_from_tokens, SentenceRows, ScoredSentenceRows
from .noun_chunks import Seq2SeqFeaturesMapperWithFocus
from .scorers import BaseScorer, Index
from .types import (
    ScholarPhiEntity,
    sentence_to_scholarphi_format,
    term_to_scholarphi_format,
    get_sentence_id
)

LOGGER = configure_logging(logger_name=__file__, logging_level=logging.INFO)


@dataclass
class Sentence(SpanGroup):
    doc: Document = None
    block: SpanGroup = None


class VilaPredTypes(IntEnum):
    paragraph: int = 0
    title: int = 1
    equation: int = 2
    reference: int = 3
    section: int = 4
    list: int = 5
    table: int = 6
    caption: int = 7
    author: int = 8
    abstract: int = 9
    footer: int = 10
    date: int = 11
    figure: int = 12


class MmdaPdfParser:
    _parser: Parser = None
    _rasterizer: Rasterizer = None
    _visual_predictor: BasePredictor = None
    _layout_predictor: BasePredictor = None
    _sentences_predictor: BasePredictor = None
    _words_predictor: BasePredictor = None
    _nlp: Seq2SeqFeaturesMapperWithFocus = None
    _scorer: BaseScorer = None

    def __init__(self, config: Optional[MmdaParserConfig] = None):
        self.config = config or MmdaParserConfig()

    @property
    def parser(self) -> Parser:
        if self._parser is None:
            self._parser = instantiate.now(self.config.parser)
        return self._parser

    @property
    def rasterizer(self) -> Rasterizer:
        if self._rasterizer is None:
            self._rasterizer = instantiate.now(self.config.rasterizer)
        return self._rasterizer

    @property
    def visual_predictor(self) -> BasePredictor:
        if self._visual_predictor is None:
            self._visual_predictor = instantiate.now(self.config.vila)
        return self._visual_predictor

    @property
    def layout_predictor(self) -> BasePredictor:
        if self._layout_predictor is None:
            self._layout_predictor = instantiate.now(self.config.lp)
        return self._layout_predictor

    @property
    def nlp(self) -> Seq2SeqFeaturesMapperWithFocus:
        if self._nlp is None:
            self._nlp = instantiate.now(self.config.noun_chunks,
                                        reuse_spacy_pipeline=True)
            # encourages caching
            self._nlp.get_spacy_pipeline()
        return self._nlp

    @property
    def sentences_predictor(self) -> BasePredictor:
        if self._sentences_predictor is None:
            self._sentences_predictor = instantiate.now(self.config.sentence)
        return self._sentences_predictor

    @property
    def words_predictor(self) -> BasePredictor:
        if self._words_predictor is None:
            self._words_predictor = instantiate.now(self.config.words)
        return self._words_predictor

    @property
    def scorer(self) -> BaseScorer:
        if self._scorer is None:
            self._scorer = instantiate.now(self.config.scorer)
        return self._scorer

    def parse_pdf(self, path: Union[str, Path]) -> Document:
        # cast if not path
        path = Path(path)
        LOGGER.info(f'Processing "{path}"...')

        # No visual prediction for now; don't need them!
        # visual_predictor = instantiate.now(self.config.vila)

        LOGGER.info(f'Parsing "{path}" with pdfplumber.')
        doc = self.parser.parse(path)

        LOGGER.info(f'Rasterizing "{path}" to images.')
        images = self.rasterizer.rasterize(input_pdf_path=path,
                                           dpi=self.config.dpi)
        doc.annotate_images(images)

        LOGGER.info(f'Predicting layout for "{path}".')
        with torch.no_grad(), warnings.catch_warnings():
            # effdet complains about __floordiv__, so we ignore warnings here
            warnings.simplefilter("ignore")
            layout_regions = self.layout_predictor.predict(doc)
        doc.annotate(blocks=layout_regions)

        words = self.words_predictor.predict(doc)
        doc.annotate(words=words)

        LOGGER.info(f'Found {len(doc.blocks)} blocks in "{path}".')

        return doc

    def get_all_sentences(self, doc: Document) -> Sequence[SentenceRows]:

        all_sentences = []

        prog = tqdm.tqdm(desc='Extracting...', unit=' sentences')
        for page in doc.pages:
            for block in page.blocks:
                block_text = [' '.join(t.symbols) for t in block.tokens]
                slices = self.sentences_predictor.\
                    split_token_based_on_sentences_boundary(block_text)

                for start, end in slices:
                    sentence_rows = slice_block_from_tokens(
                        block=block,
                        bos_token=block.tokens[start],
                        eos_token=block.tokens[end - 1])

                    all_sentences.append(sentence_rows)
                    prog.update()
        prog.close()

        LOGGER.info(f'Extracted {prog.n:,} sentences.')

        return all_sentences

    def get_abstract_and_body(
        self,
        extracted_sentences: Sequence[SentenceRows],
        title_label: str = 'Title',
        text_label: str = 'Text'
    ) -> Tuple[Sequence[SentenceRows], Sequence[SentenceRows]]:
        in_abstract = False
        abstract, body = [], []

        for sentence in extracted_sentences:
            if sentence.type == title_label:
                in_abstract = \
                    sentence.sentence.lower().replace(' ', '') == 'abstract'

            elif sentence.type == text_label:
                (abstract if in_abstract else body).append(sentence)

        return abstract, body

    def get_noun_chunk(
        self,
        abstract: Sequence[SentenceRows]
    ) -> Sequence[SpanGroup]:

        abstract_noun_chunks = []
        prog = tqdm.tqdm(desc='Getting Noun Chunks from abstract...',
                         unit=' NCs')

        for sentence in abstract:
            sentence_words = sentence.words
            block_text = Doc(self.nlp.get_spacy_pipeline().vocab,
                             [' '.join(w.symbols) for w in sentence_words])
            parsed_sentence = self.nlp.get_spacy_pipeline()(block_text)

            for nc in self.nlp.find_chunks(parsed_sentence):
                print(repr(str(nc)))
                noun_chunk = slice_block_from_tokens(
                    block=sentence,
                    bos_token=sentence_words[nc.start],
                    # positions in spacy are exclusive of last position,
                    # but not in our code! So "-1"  it is.
                    eos_token=sentence_words[nc.end - 1]
                )

                abstract_noun_chunks.append(noun_chunk)
                prog.update(1)

        return abstract_noun_chunks

    def score_body_sentences(
        self,
        noun_chunk: SpanGroup,
        body: Sequence[SentenceRows],
        index: Index
    ) -> Sequence[ScoredSentenceRows]:
        all_scored_sentence = []

        for sentence in body:
            score = self.scorer.score(
                query=noun_chunk, text=sentence, index=index
            )
            scored_sentence = ScoredSentenceRows.from_sentence(
                score=score, query=noun_chunk, sentence=sentence
            )
            all_scored_sentence.append(scored_sentence)

        return all_scored_sentence

    def score_all_sentences(
        self,
        path: Union[str, Path]
    ) -> Sequence[ScholarPhiEntity]:

        doc = self.parse_pdf(path)
        sentences = self.get_all_sentences(doc)
        index = self.scorer.build_index(sentences)
        abstract, body = self.get_abstract_and_body(sentences)
        noun_chunks = self.get_noun_chunk(abstract)

        entities = OrderedDict()

        for nc in noun_chunks:

            scored_sentences = self.score_body_sentences(
                noun_chunk=nc, body=body, index=index
            )
            scored_sentences_entities = [
                entities.setdefault(get_sentence_id(s),
                                    sentence_to_scholarphi_format(s))
                for s in scored_sentences
            ]
            [
                entities.setdefault(ent.id, ent)
                for ent in term_to_scholarphi_format(
                    span=nc,
                    sentences=scored_sentences_entities,
                    scores=(s.score for s in scored_sentences)
                )
            ]

        return tuple(entities.values())



@cli(CliMmdaParserConfig)
def main(config: CliMmdaParserConfig):
    configure_logging(logging_level=config.logging_level)
    parser = MmdaPdfParser(config)

    parsed = parser.score_all_sentences(path=config.path)

    if config.save_path:
        with open(config.save_path, 'w', encoding='utf-8') as f:
            for sentence in parsed:
                f.write(sentence.json() + '\n')


if __name__ == '__main__':
    main()
