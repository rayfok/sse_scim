
from dataclasses import dataclass
from enum import IntEnum
import json
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
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
from .utils import slice_block_from_tokens, SentenceRows
from .noun_chunks import Seq2SeqFeaturesMapperWithFocus

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
        for sentence in abstract:
            block_text = Doc(self.nlp.get_spacy_pipeline().vocab,
                             [' '.join(w.symbols) for w in sentence.words])

            parsed_sentence = self.nlp.get_spacy_pipeline()(sentence.sentence)

            # for nc in self.nlp.find_chunks(parsed_sentence):
            #     slice_block_from_tokens(block=sentence, bos_token=sentence.tokens)


            import ipdb
            ipdb.set_trace()

    def score_body_sentences(
        self,
        noun_chunk: SpanGroup,
        body: Sequence[SentenceRows]
    ) -> Sequence[SpanGroup]:
        ...

    def __call__(self, path: Union[str, Path]):
        doc = self.parse_pdf(path)
        sentences = self.get_all_sentences(doc)
        abstract, body = self.get_abstract_and_body(sentences)
        noun_chunks = self.get_noun_chunk(abstract)
        scored_body = {nc: self.score_body_sentences(noun_chunk=nc, body=body)
                       for nc in noun_chunks}

        return scored_body


@cli(CliMmdaParserConfig)
def main(config: CliMmdaParserConfig):
    configure_logging(logging_level=config.logging_level)
    parser = MmdaPdfParser(config)

    parsed = parser(path=config.path)
    if config.save_path:
        with open(config.save_path, 'w', encoding='utf-8') as f:
            for sentence in parsed:
                f.write(json.dumps(sentence.to_json()) + '\n')


if __name__ == '__main__':
    main()
