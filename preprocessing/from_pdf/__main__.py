
from pathlib import Path
import warnings
from dataclasses import dataclass

import torch
from cached_path import cached_path

from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.predictors.heuristic_predictors.sentence_boundary_predictor import \
    PysbdSentenceBoundaryPredictor
from mmda.predictors.lp_predictors import LayoutParserPredictor
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer

import springs as sp

from .typed_predictors import TypedBlockPredictor, TypedSentencesPredictor
from .word_predictors import ExtendedDictionaryWordPredictor
from .visualization import visualize_typed_sentences    # noqa: F401


WORDS_URL = 'https://github.com/dwyl/english-words/raw/master/words.txt'


@sp.make_flexy
@dataclass
class PipelineStepConfig:
    _target_: str = sp.MISSING


@dataclass
class PipelineConfig:
    parser: PipelineStepConfig = PipelineStepConfig(
        _target_=sp.Target.to_string(PDFPlumberParser)
    )
    rasterizer: PipelineStepConfig = PipelineStepConfig(
        _target_=sp.Target.to_string(PDF2ImageRasterizer)
    )
    layout: PipelineStepConfig = sp.flexy_field(
        PipelineStepConfig,
        _target_=sp.Target.to_string(LayoutParserPredictor.from_pretrained),
        config_path="lp://efficientdet/PubLayNet"
    )
    sents: PipelineStepConfig = PipelineStepConfig(
        _target_=sp.Target.to_string(PysbdSentenceBoundaryPredictor)
    )
    words: PipelineStepConfig = sp.flexy_field(
        PipelineStepConfig,
        _target_=sp.Target.to_string(ExtendedDictionaryWordPredictor),
        dictionary_file_path=str(cached_path(WORDS_URL))
    )
    blocks_type: PipelineStepConfig = PipelineStepConfig(
        _target_=sp.Target.to_string(TypedBlockPredictor)
    )
    sents_type: PipelineStepConfig = PipelineStepConfig(
        _target_=sp.Target.to_string(TypedSentencesPredictor)
    )


@sp.dataclass
class AppConfig:
    pipeline: PipelineConfig = PipelineConfig()
    src: str = sp.MISSING
    dst: str = sp.MISSING


@sp.cli(AppConfig)
def main(config: AppConfig):
    path = Path(cached_path(config.src))

    parser = sp.init.now(config.pipeline.parser,
                         PDFPlumberParser)
    rasterizer = sp.init.now(config.pipeline.rasterizer,
                             PDF2ImageRasterizer)
    layout_pred = sp.init.now(config.pipeline.layout,
                              LayoutParserPredictor)
    words_pred = sp.init.now(config.pipeline.words,
                             ExtendedDictionaryWordPredictor)
    sents_pred = sp.init.now(config.pipeline.sents,
                             PysbdSentenceBoundaryPredictor)
    blocks_type_pred = sp.init.now(config.pipeline.blocks_type,
                                   TypedBlockPredictor)
    sents_type_pred = sp.init.now(config.pipeline.sents_type,
                                  TypedSentencesPredictor)

    doc = parser.parse(str(path))
    images = rasterizer.rasterize(str(path), dpi=72)
    doc.annotate_images(images)

    with torch.no_grad(), warnings.catch_warnings():
        layout_regions = layout_pred.predict(doc)
        doc.annotate(blocks=layout_regions)

    words = words_pred.predict(doc)
    doc.annotate(words=words)

    sents = sents_pred.predict(doc)
    doc.annotate(sents=sents)

    typed_blocks = blocks_type_pred.predict(doc)
    doc.annotate(typed_blocks=typed_blocks)

    typed_sents = sents_type_pred.predict(doc)
    doc.annotate(typed_sents=typed_sents)

    visualize_typed_sentences(doc, config.dst)


if __name__ == '__main__':
    main()
